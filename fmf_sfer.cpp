#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <limits>
#include <unordered_set>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

template <class T>
using vec = std::vector<T>;

// ============================================================
// Индексация состояния
// 0..5   : x, y, z, u, v, w
// 6..14  : J11..J33
// 15..23 : L11..L33, где L = dJ/dt
// ============================================================

static constexpr int NVAR = 24;

static constexpr int IX = 0;
static constexpr int IY = 1;
static constexpr int IZ = 2;
static constexpr int IU = 3;
static constexpr int IV = 4;
static constexpr int IW = 5;

static constexpr int IJ = 6;
static constexpr int IL = 15;

inline int Jidx(int i, int j) { return IJ + 3 * i + j; }
inline int Lidx(int i, int j) { return IL + 3 * i + j; }

static constexpr double PI = 3.1415926535897932384626433832795;

// ============================================================
// Параметры модели
// ============================================================

static constexpr double DELTA = 1.0;
static constexpr double ALPHA = 1.1;
static constexpr double KAPPA = 0.7;

static constexpr double BETA1  = 0.5;
static constexpr double A_SPEED = 1.0;

// ============================================================
// Параметры центрального ядра
// ============================================================

static constexpr double CORE_RADIUS = 0.1;
static constexpr double CORE_RADIUS2 = CORE_RADIUS * CORE_RADIUS;

// ============================================================
// Защита от слишком малых |detJ|
// ============================================================

static constexpr double DETJ_EPS = 1e-14;

// ============================================================
// Защита для малых скоростей и малых S
// ============================================================

static constexpr double V_EPS = 1e-12;
static constexpr double S_EPS = 1e-6;

// ============================================================
// Специальное исключение: частица упала на ядро
// ============================================================

struct ParticleHitCore : public std::runtime_error {
    ParticleHitCore() : std::runtime_error("Particle reached core radius") {}
};

// ============================================================
// Определитель матрицы J
// ============================================================

double det3x3_from_state(const vec<double>& y)
{
    const double J11 = y[Jidx(0,0)], J12 = y[Jidx(0,1)], J13 = y[Jidx(0,2)];
    const double J21 = y[Jidx(1,0)], J22 = y[Jidx(1,1)], J23 = y[Jidx(1,2)];
    const double J31 = y[Jidx(2,0)], J32 = y[Jidx(2,1)], J33 = y[Jidx(2,2)];

    return
        J11 * (J22 * J33 - J23 * J32)
      - J12 * (J21 * J33 - J23 * J31)
      + J13 * (J21 * J32 - J22 * J31);
}

// ============================================================
// Cd(S), beta(S) и d beta / dV
// ============================================================

double Cd(double S)
{
    const double absS = std::fabs(S);

    if (absS < S_EPS) {
        const double inv_sqrt_pi = 1.0 / std::sqrt(PI);
        const double S2 = S * S;
        return inv_sqrt_pi * (
            16.0 / (3.0 * S)
          + 16.0 * S / 15.0
          - 8.0 * S * S2 / 105.0
        );
    }

    const double S2 = S * S;
    const double S3 = S2 * S;
    const double S4 = S2 * S2;

    return
        ((4.0 * S4 + 4.0 * S2 - 1.0) / (2.0 * S4)) * std::erf(S)
      + ((2.0 * S2 + 1.0) / (std::sqrt(PI) * S3)) * std::exp(-S2);
}

// F(S) = S Cd(S). Именно beta = beta1 F(S).
double S_times_Cd(double S)
{
    const double absS = std::fabs(S);

    if (absS < S_EPS) {
        const double inv_sqrt_pi = 1.0 / std::sqrt(PI);
        const double S2 = S * S;
        const double S4 = S2 * S2;
        const double S6 = S4 * S2;
        return inv_sqrt_pi * (
            16.0 / 3.0
          + 16.0 * S2 / 15.0
          - 8.0 * S4 / 105.0
          + 8.0 * S6 / 945.0
        );
    }

    return S * Cd(S);
}

// d/dS [S Cd(S)] = Cd(S) + S Cd'(S).
double d_S_times_Cd_dS(double S)
{
    const double absS = std::fabs(S);

    if (absS < S_EPS) {
        const double inv_sqrt_pi = 1.0 / std::sqrt(PI);
        const double S2 = S * S;
        const double S3 = S2 * S;
        const double S5 = S3 * S2;
        return inv_sqrt_pi * (
            32.0 * S / 15.0
          - 32.0 * S3 / 105.0
          + 16.0 * S5 / 315.0
        );
    }

    const double S2 = S * S;
    const double S3 = S2 * S;
    const double S4 = S2 * S2;

    return
        (2.0 - 2.0 / S2 + 3.0 / (2.0 * S4)) * std::erf(S)
      + ((2.0 * S2 - 3.0) / (std::sqrt(PI) * S3)) * std::exp(-S2);
}

struct Coefficients {
    double r = 0.0;
    double vabs = 0.0;
    double S = 0.0;
    double a = 0.0;
    double c = 0.0;
    double beta = 0.0;
    double dbeta_dV = 0.0;
};

Coefficients compute_coefficients(double x, double y, double z,
                                  double u, double v, double w)
{
    Coefficients q;

    const double r2 = x * x + y * y + z * z;
    q.r = std::sqrt(r2);

    const double v2 = u * u + v * v + w * w;
    q.vabs = std::sqrt(v2);

    q.S = A_SPEED * q.vabs;

    // alpha = alpha_0 + kappa/r^3; a = delta*alpha, c = delta*(alpha-1)
    const double alpha_total = ALPHA + KAPPA / (r2 * q.r);
    q.a = DELTA * alpha_total;
    q.c = DELTA * (alpha_total - 1.0);

    q.beta = BETA1 * S_times_Cd(q.S);
    q.dbeta_dV = BETA1 * A_SPEED * d_S_times_Cd_dS(q.S);

    return q;
}

// ============================================================
// Правая часть системы
// ============================================================

vec<double> f(const vec<double>& y)
{
    vec<double> dy(NVAR, 0.0);

    const double x  = y[IX];
    const double yy = y[IY];
    const double z  = y[IZ];
    const double u  = y[IU];
    const double v  = y[IV];
    const double w  = y[IW];

    const double r2 = x*x + yy*yy + z*z;

    if (r2 <= CORE_RADIUS2) {
        throw ParticleHitCore();
    }

    const double r = std::sqrt(r2);
    const double r5 = r2 * r2 * r;

    const Coefficients q = compute_coefficients(x, yy, z, u, v, w);

    dy[IX] = u;
    dy[IY] = v;
    dy[IZ] = w;

    dy[IU] = -q.beta * u - q.c * x  + 2.0 * v;
    dy[IV] = -q.beta * v - q.c * yy - 2.0 * u;
    dy[IW] = -q.beta * w - q.a * z;

    for (int j = 0; j < 3; ++j) {
        const double J1j = y[Jidx(0,j)];
        const double J2j = y[Jidx(1,j)];
        const double J3j = y[Jidx(2,j)];

        const double L1j = y[Lidx(0,j)];
        const double L2j = y[Lidx(1,j)];
        const double L3j = y[Lidx(2,j)];

        dy[Jidx(0,j)] = L1j;
        dy[Jidx(1,j)] = L2j;
        dy[Jidx(2,j)] = L3j;

        const double pos_dot_J = x * J1j + yy * J2j + z * J3j;
        // da/dx0j = dc/dx0j = delta*d(alpha)/dx0j
        const double da_dx0j = -3.0 * DELTA * KAPPA * pos_dot_J / r5;
        const double dc_dx0j = da_dx0j;

        double dbeta_dx0j = 0.0;
        if (q.vabs > V_EPS) {
            const double vel_dot_L = u * L1j + v * L2j + w * L3j;
            dbeta_dx0j = q.dbeta_dV * vel_dot_L / q.vabs;
        }

        dy[Lidx(0,j)] =
            -q.beta * L1j
            -q.c * J1j
            + 2.0 * L2j
            - u * dbeta_dx0j
            - x * dc_dx0j;

        dy[Lidx(1,j)] =
            -q.beta * L2j
            -q.c * J2j
            - 2.0 * L1j
            - v * dbeta_dx0j
            - yy * dc_dx0j;

        dy[Lidx(2,j)] =
            -q.beta * L3j
            -q.a * J3j
            - w * dbeta_dx0j
            - z * da_dx0j;
    }

    return dy;
}

// ============================================================
// Норма ошибки
// ============================================================

double error_norm(const vec<double>& err,
                  const vec<double>& y,
                  const vec<double>& y_new,
                  double atol,
                  double rtol)
{
    double s = 0.0;
    size_t n = err.size();

    for (size_t i = 0; i < n; ++i) {
        double scale = atol + rtol * std::max(std::fabs(y[i]), std::fabs(y_new[i]));
        double e = err[i] / scale;
        s += e * e;
    }

    return std::sqrt(s / std::max<size_t>(1, n));
}

// ============================================================
// Линейная комбинация векторов
// ============================================================

void lin_comb(vec<double>& out,
              const std::initializer_list<std::pair<double, const vec<double>*>>& terms)
{
    const size_t n = out.size();
    for (size_t i = 0; i < n; ++i) out[i] = 0.0;

    for (const auto& term : terms) {
        const double c = term.first;
        const vec<double>& v = *term.second;
        for (size_t i = 0; i < n; ++i) {
            out[i] += c * v[i];
        }
    }
}

// ============================================================
// Сетка концентрации 2D
// ============================================================

struct ConcentrationGrid2D {
    int nx, ny;
    double xmin, xmax, ymin, ymax;

    vec<double> sum_c;
    vec<double> sum_c2;
    vec<long long> hits;

    ConcentrationGrid2D(int nx_, int ny_,
                        double xmin_, double xmax_,
                        double ymin_, double ymax_)
        : nx(nx_), ny(ny_),
          xmin(xmin_), xmax(xmax_),
          ymin(ymin_), ymax(ymax_),
          sum_c(static_cast<size_t>(nx_) * ny_, 0.0),
          sum_c2(static_cast<size_t>(nx_) * ny_, 0.0),
          hits(static_cast<size_t>(nx_) * ny_, 0)
    {}

    int idx(int ix, int iy) const {
        return iy * nx + ix;
    }

    bool add_sample(double x, double y, double c) {
        if (x < xmin || x >= xmax || y < ymin || y >= ymax) return false;

        const double tx = (x - xmin) / (xmax - xmin);
        const double ty = (y - ymin) / (ymax - ymin);

        int ix = static_cast<int>(tx * nx);
        int iy = static_cast<int>(ty * ny);

        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return false;

        const int k = idx(ix, iy);
        sum_c[k] += c;
        sum_c2[k] += c * c;
        hits[k] += 1;
        return true;
    }

    void save_csv(const std::string& filename,
                  const std::string& x_name,
                  const std::string& y_name) const
    {
        std::ofstream fout(filename);
        if (!fout) {
            throw std::runtime_error("Cannot open grid output file: " + filename);
        }

        fout << std::setprecision(16);
        fout << "ix,iy," << x_name << "_center," << y_name
             << "_center,c_mean,c_std,c_sum,hits\n";

        const double dx = (xmax - xmin) / nx;
        const double dy = (ymax - ymin) / ny;

        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const int k = idx(ix, iy);

                const double xc = xmin + (ix + 0.5) * dx;
                const double yc = ymin + (iy + 0.5) * dy;

                double mean = 0.0;
                double stddev = 0.0;

                if (hits[k] > 0) {
                    mean = sum_c[k] / hits[k];
                    const double mean2 = sum_c2[k] / hits[k];
                    const double var = std::max(0.0, mean2 - mean * mean);
                    stddev = std::sqrt(var);
                }

                fout << ix << ","
                     << iy << ","
                     << xc << ","
                     << yc << ","
                     << mean << ","
                     << stddev << ","
                     << sum_c[k] << ","
                     << hits[k] << "\n";
            }
        }
    }
};

// ============================================================
// Начальное объёмное облако: набор сферических слоёв
// ============================================================

struct ParticleIC {
    double x0, y0, z0;
};

std::vector<ParticleIC> generate_ball_cloud(int n_r,
                                            int n_theta,
                                            int n_phi,
                                            double r_min,
                                            double r_max)
{
    std::vector<ParticleIC> pts;
    pts.reserve(static_cast<size_t>(n_r) * n_theta * n_phi);

    for (int ir = 0; ir < n_r; ++ir) {
        const double q = (n_r == 1)
            ? 0.0
            : static_cast<double>(ir) / (n_r - 1);

        // Равномернее по объёму, чем линейный радиус.
        const double radius = std::cbrt(
            r_min * r_min * r_min +
            q * (r_max * r_max * r_max - r_min * r_min * r_min)
        );

        for (int it = 0; it < n_theta; ++it) {
            const double mu = (n_theta == 1)
                ? 0.0
                : -1.0 + 2.0 * static_cast<double>(it) / (n_theta - 1);

            const double theta = std::acos(std::clamp(mu, -1.0, 1.0));
            const double st = std::sin(theta);
            const double ct = std::cos(theta);

            for (int ip = 0; ip < n_phi; ++ip) {
                const double phi = 2.0 * PI * static_cast<double>(ip) / n_phi;

                ParticleIC p;
                p.x0 = radius * st * std::cos(phi);
                p.y0 = radius * st * std::sin(phi);
                p.z0 = radius * ct;
                pts.push_back(p);
            }
        }
    }

    return pts;
}

// ============================================================
// DOPRI8
// ============================================================

struct Dopri8 {
    double atol  = 1e-12;
    double rtol  = 1e-10;

    double safety     = 0.9;
    double grow_max   = 5.0;
    double shrink_min = 0.2;

    double h_min = 1e-12;
    double h_max = 0.1;

    static constexpr int p = 8;

    struct Result {
        vec<double> yT;
        long long accepted_steps = 0;
        long long rejected_steps = 0;
        bool hit_core = false;
        double t_end = 0.0;
    };

    template <class Observer>
    Result integrate(double T,
                     double h0,
                     const vec<double>& y0,
                     Observer&& observer,
                     int observe_every_accepted_step = 1) const
    {
        vec<double> y = y0;

        double t = 0.0;
        double h = std::clamp(h0, h_min, h_max);

        vec<double> k1(NVAR), k2(NVAR), k3(NVAR), k4(NVAR), k5(NVAR), k6(NVAR),
                    k7(NVAR), k8(NVAR), k9(NVAR), k10(NVAR), k11(NVAR), k12(NVAR);
        vec<double> yt(NVAR), y_high(NVAR), errv(NVAR);

        long long accepted_steps = 0;
        long long rejected_steps = 0;

        observer(t, y);

        const double b1  = 5.42937341165687622380535766363e-2;
        const double b6  = 4.45031289275240888144113950566;
        const double b7  = 1.89151789931450038304281599044;
        const double b8  = -5.8012039600105847814672114227;
        const double b9  = 3.1116436695781989440891606237e-1;
        const double b10 = -1.52160949662516078556178806805e-1;
        const double b11 = 2.01365400804030348374776537501e-1;
        const double b12 = 4.47106157277725905176885569043e-2;

        const double er1  = 0.1312004499419488073250102996e-01;
        const double er6  = -0.1225156446376204440720569753e+01;
        const double er7  = -0.4957589496572501915214079952;
        const double er8  = 0.1664377182454986536961530415e+01;
        const double er9  = -0.3503288487499736816886487290;
        const double er10 = 0.3341791187130174790297318841;
        const double er11 = 0.8192320648511571246570742613e-01;
        const double er12 = -0.2235530786388629525884427845e-01;

        const double a21   = 5.26001519587677318785587544488e-2;
        const double a31   = 1.97250569845378994544595329183e-2;
        const double a32   = 5.91751709536136983633785987549e-2;
        const double a41   = 2.95875854768068491816892993775e-2;
        const double a43   = 8.87627564304205475450678981324e-2;
        const double a51   = 2.41365134159266685502369798665e-1;
        const double a53   = -8.84549479328286085344864962717e-1;
        const double a54   = 9.24834003261792003115737966543e-1;
        const double a61   = 3.7037037037037037037037037037e-2;
        const double a64   = 1.70828608729473871279604482173e-1;
        const double a65   = 1.25467687566822425016691814123e-1;
        const double a71   = 3.7109375e-2;
        const double a74   = 1.70252211019544039314978060272e-1;
        const double a75   = 6.02165389804559606850219397283e-2;
        const double a76   = -1.7578125e-2;
        const double a81   = 3.70920001185047927108779319836e-2;
        const double a84   = 1.70383925712239993810214054705e-1;
        const double a85   = 1.07262030446373284651809199168e-1;
        const double a86   = -1.53194377486244017527936158236e-2;
        const double a87   = 8.27378916381402288758473766002e-3;
        const double a91   = 6.24110958716075717114429577812e-1;
        const double a94   = -3.36089262944694129406857109825;
        const double a95   = -8.68219346841726006818189891453e-1;
        const double a96   = 2.75920996994467083049415600797e+1;
        const double a97   = 2.01540675504778934086186788979e+1;
        const double a98   = -4.34898841810699588477366255144e+1;
        const double a101  = 4.77662536438264365890433908527e-1;
        const double a104  = -2.48811461997166764192642586468;
        const double a105  = -5.90290826836842996371446475743e-1;
        const double a106  = 2.12300514481811942347288949897e+1;
        const double a107  = 1.52792336328824235832596922938e+1;
        const double a108  = -3.32882109689848629194453265587e+1;
        const double a109  = -2.03312017085086261358222928593e-2;
        const double a111  = -9.3714243008598732571704021658e-1;
        const double a114  = 5.18637242884406370830023853209;
        const double a115  = 1.09143734899672957818500254654;
        const double a116  = -8.14978701074692612513997267357;
        const double a117  = -1.85200656599969598641566180701e+1;
        const double a118  = 2.27394870993505042818970056734e+1;
        const double a119  = 2.49360555267965238987089396762;
        const double a1110 = -3.0467644718982195003823669022;
        const double a121  = 2.27331014751653820792359768449;
        const double a124  = -1.05344954667372501984066689879e+1;
        const double a125  = -2.00087205822486249909675718444;
        const double a126  = -1.79589318631187989172765950534e+1;
        const double a127  = 2.79488845294199600508499808837e+1;
        const double a128  = -2.85899827713502369474065508674;
        const double a129  = -8.87285693353062954433549289258;
        const double a1210 = 1.23605671757943030647266201528e+1;
        const double a1211 = 6.43392746015763530355970484046e-1;

        try {
            while (t < T) {
                if (t + h > T) h = T - t;

                k1 = f(y);

                lin_comb(yt, { {1.0, &y}, {h * a21, &k1} });
                k2 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a31, &k1}, {h * a32, &k2} });
                k3 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a41, &k1}, {h * a43, &k3} });
                k4 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a51, &k1}, {h * a53, &k3}, {h * a54, &k4} });
                k5 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a61, &k1}, {h * a64, &k4}, {h * a65, &k5} });
                k6 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a71, &k1}, {h * a74, &k4}, {h * a75, &k5}, {h * a76, &k6} });
                k7 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a81, &k1}, {h * a84, &k4}, {h * a85, &k5}, {h * a86, &k6}, {h * a87, &k7} });
                k8 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a91, &k1}, {h * a94, &k4}, {h * a95, &k5}, {h * a96, &k6}, {h * a97, &k7}, {h * a98, &k8} });
                k9 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a101, &k1}, {h * a104, &k4}, {h * a105, &k5}, {h * a106, &k6}, {h * a107, &k7}, {h * a108, &k8}, {h * a109, &k9} });
                k10 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a111, &k1}, {h * a114, &k4}, {h * a115, &k5}, {h * a116, &k6}, {h * a117, &k7}, {h * a118, &k8}, {h * a119, &k9}, {h * a1110, &k10} });
                k11 = f(yt);

                lin_comb(yt, { {1.0, &y}, {h * a121, &k1}, {h * a124, &k4}, {h * a125, &k5}, {h * a126, &k6}, {h * a127, &k7}, {h * a128, &k8}, {h * a129, &k9}, {h * a1210, &k10}, {h * a1211, &k11} });
                k12 = f(yt);

                lin_comb(y_high, {
                    {1.0, &y},
                    {h * b1,  &k1},
                    {h * b6,  &k6},
                    {h * b7,  &k7},
                    {h * b8,  &k8},
                    {h * b9,  &k9},
                    {h * b10, &k10},
                    {h * b11, &k11},
                    {h * b12, &k12}
                });

                lin_comb(errv, {
                    {h * er1,  &k1},
                    {h * er6,  &k6},
                    {h * er7,  &k7},
                    {h * er8,  &k8},
                    {h * er9,  &k9},
                    {h * er10, &k10},
                    {h * er11, &k11},
                    {h * er12, &k12}
                });

                const double E = error_norm(errv, y, y_high, atol, rtol);

                double fac = safety * std::pow(std::max(E, 1e-16), -1.0 / (p + 1));
                fac = std::clamp(fac, shrink_min, grow_max);
                const double h_new = std::clamp(h * fac, h_min, h_max);

                if (E <= 1.0) {
                    y = y_high;
                    t += h;
                    ++accepted_steps;
                    h = h_new;

                    if (accepted_steps % observe_every_accepted_step == 0 || t >= T) {
                        observer(t, y);
                    }
                } else {
                    ++rejected_steps;
                    h = h_new;

                    if (h <= h_min * 1.0001) {
                        throw std::runtime_error("Dopri8: step became too small");
                    }
                }
            }
        }
        catch (const ParticleHitCore&) {
            Result R;
            R.yT = y;
            R.accepted_steps = accepted_steps;
            R.rejected_steps = rejected_steps;
            R.hit_core = true;
            R.t_end = t;
            return R;
        }

        Result R;
        R.yT = y;
        R.accepted_steps = accepted_steps;
        R.rejected_steps = rejected_steps;
        R.hit_core = false;
        R.t_end = t;
        return R;
    }
};

// ============================================================
// Точка траектории / интерполяции
// ============================================================

struct SamplePoint {
    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double r = 0.0;
    double u = 0.0;
    double v = 0.0;
    double w = 0.0;

    double detJ_signed = 0.0;
    double detJ_abs = 0.0;
    double c = 0.0;
    bool valid_c = false;
};

SamplePoint make_sample(double t, const vec<double>& y)
{
    SamplePoint s;
    s.t = t;
    s.x = y[IX];
    s.y = y[IY];
    s.z = y[IZ];
    s.r = std::sqrt(s.x * s.x + s.y * s.y);
    s.u = y[IU];
    s.v = y[IV];
    s.w = y[IW];

    s.detJ_signed = det3x3_from_state(y);
    s.detJ_abs = std::fabs(s.detJ_signed);
    s.valid_c = (s.detJ_abs > DETJ_EPS);
    s.c = s.valid_c ? (1.0 / s.detJ_abs) : 0.0;
    return s;
}

SamplePoint lerp_sample(const SamplePoint& a, const SamplePoint& b, double s)
{
    SamplePoint p;
    p.t = a.t + s * (b.t - a.t);
    p.x = a.x + s * (b.x - a.x);
    p.y = a.y + s * (b.y - a.y);
    p.z = a.z + s * (b.z - a.z);
    p.r = a.r + s * (b.r - a.r);
    p.u = a.u + s * (b.u - a.u);
    p.v = a.v + s * (b.v - a.v);
    p.w = a.w + s * (b.w - a.w);

    p.detJ_signed = a.detJ_signed + s * (b.detJ_signed - a.detJ_signed);
    p.detJ_abs = std::fabs(p.detJ_signed);
    p.valid_c = (p.detJ_abs > DETJ_EPS);
    p.c = p.valid_c ? (1.0 / p.detJ_abs) : 0.0;
    return p;
}

bool has_sign_change(double a, double b)
{
    return (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0);
}

// ============================================================
// Начальные точки стационарного влёта: фиксированная сфера r = R0
// ============================================================

std::vector<ParticleIC> generate_fixed_sphere(int n_theta,
                                              int n_phi,
                                              double radius,
                                              bool upper_hemisphere_only)
{
    std::vector<ParticleIC> pts;
    pts.reserve(static_cast<size_t>(n_theta) * n_phi);

    const double mu_min = upper_hemisphere_only ? 0.0 : -1.0;
    const double mu_max = 1.0;

    for (int it = 0; it < n_theta; ++it) {
        const double q = (n_theta == 1) ? 0.5 : static_cast<double>(it) / (n_theta - 1);
        const double mu = mu_min + (mu_max - mu_min) * q;
        const double theta = std::acos(std::clamp(mu, -1.0, 1.0));
        const double st = std::sin(theta);
        const double ct = std::cos(theta);

        for (int ip = 0; ip < n_phi; ++ip) {
            const double phi = 2.0 * PI * static_cast<double>(ip) / n_phi;
            ParticleIC p;
            p.x0 = radius * st * std::cos(phi);
            p.y0 = radius * st * std::sin(phi);
            p.z0 = radius * ct;
            pts.push_back(p);
        }
    }
    return pts;
}

// ============================================================
// main
// ============================================================

int main()
{
    try {
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(12);

        // ============================================================
        // СТАЦИОНАРНАЯ ЗАДАЧА
        // Частицы непрерывно влетают с одной фиксированной сферы.
        // Поле строится как совокупность точек разных возрастов tau.
        // ============================================================

        const double INJECTION_RADIUS = 1.0;
        const bool UPPER_HEMISPHERE_ONLY = false;

        // Сетка начальных точек на фиксированной сфере r = INJECTION_RADIUS.
        const int n_theta = 120;
        const int n_phi = 200;

        std::vector<ParticleIC> cloud =
            generate_fixed_sphere(n_theta, n_phi, INJECTION_RADIUS, UPPER_HEMISPHERE_ONLY);

        std::cout << "Stationary boundary inflow problem\n";
        std::cout << "Injection points: " << cloud.size() << "\n";
        std::cout << "Injection sphere radius: " << INJECTION_RADIUS << "\n";
        std::cout << "Core radius: " << CORE_RADIUS << "\n";
        std::cout << "DELTA=" << DELTA
                  << ", ALPHA0=" << ALPHA
                  << ", KAPPA=" << KAPPA
                  << ", BETA1=" << BETA1
                  << ", A_SPEED=" << A_SPEED << "\n";

        // Максимальный возраст частиц в стационарном ансамбле.
        const double T_AGE_MAX = 20.0 * PI;
        const double h0 = 1e-3;

        Dopri8 solver;
        solver.atol = 1e-12;
        solver.rtol = 1e-10;
        solver.h_min = 1e-12;
        solver.h_max = 0.05;

        // Для поля концентрации в плоскости r-z.
        ConcentrationGrid2D rz_grid(700, 700, 0.0, 1.5, -1.5, 1.5);

        // Каустика на экваторе: сохраняем точки detJ=0, лежащие в тонком слое |z| <= band.
        const double EQUATOR_CAUSTIC_BAND = 2.0e-3;

        std::ofstream caustics_rz("caustic_points_rz.csv");
        std::ofstream caustics_equator("caustic_points_equator.csv");

        if (!caustics_rz) throw std::runtime_error("Cannot open caustic_points_rz.csv");
        if (!caustics_equator) throw std::runtime_error("Cannot open caustic_points_equator.csv");

        caustics_rz << std::setprecision(16);
        caustics_equator << std::setprecision(16);

        caustics_rz << "particle_id,age_caus,x,y,z,r,detJ_prev,detJ_next\n";
        caustics_equator << "particle_id,age_caus,x,y,z,r,detJ_prev,detJ_next\n";

        long long total_accepted = 0;
        long long total_rejected = 0;
        long long particles_hit_core = 0;
        long long particles_failed = 0;
        long long particles_completed = 0;
        long long total_samples = 0;
        long long total_caustics_rz = 0;
        long long total_caustics_equator = 0;

        const int observe_every = 1;

        for (size_t pid = 0; pid < cloud.size(); ++pid) {
            vec<double> y0(NVAR, 0.0);
            y0[IX] = cloud[pid].x0;
            y0[IY] = cloud[pid].y0;
            y0[IZ] = cloud[pid].z0;
            y0[IU] = 0.0;
            y0[IV] = 0.0;
            y0[IW] = 0.0;

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    y0[Jidx(i,j)] = (i == j ? 1.0 : 0.0);
                    y0[Lidx(i,j)] = 0.0;
                }
            }

            bool have_prev = false;
            SamplePoint prev;

            auto observer = [&](double t, const vec<double>& y) {
                const SamplePoint s = make_sample(t, y);

                // Стационарное поле концентрации в r-z.
                // Сохраняем только итоговую сетку, без облака sample-points.
                if (s.valid_c) {
                    rz_grid.add_sample(s.r, s.z, s.c);
                    ++total_samples;
                }

                if (have_prev) {
                    // Каустики: detJ меняет знак, значит между соседними шагами был detJ=0.
                    if (has_sign_change(prev.detJ_signed, s.detJ_signed)) {
                        const double denom = s.detJ_signed - prev.detJ_signed;
                        if (std::fabs(denom) > 0.0) {
                            const double q = -prev.detJ_signed / denom;
                            if (q >= 0.0 && q <= 1.0) {
                                const SamplePoint cp = lerp_sample(prev, s, q);

                                caustics_rz
                                    << pid << "," << cp.t << ","
                                    << cp.x << "," << cp.y << "," << cp.z << "," << cp.r << ","
                                    << prev.detJ_signed << "," << s.detJ_signed << "\n";
                                ++total_caustics_rz;

                                if (std::fabs(cp.z) <= EQUATOR_CAUSTIC_BAND) {
                                    caustics_equator
                                        << pid << "," << cp.t << ","
                                        << cp.x << "," << cp.y << "," << cp.z << "," << cp.r << ","
                                        << prev.detJ_signed << "," << s.detJ_signed << "\n";
                                    ++total_caustics_equator;
                                }
                            }
                        }
                    }
                }

                prev = s;
                have_prev = true;
            };

            try {
                auto Rres = solver.integrate(T_AGE_MAX, h0, y0, observer, observe_every);
                total_accepted += Rres.accepted_steps;
                total_rejected += Rres.rejected_steps;

                if (Rres.hit_core) ++particles_hit_core;
                else ++particles_completed;
            }
            catch (const std::exception& e) {
                ++particles_failed;
                std::cerr << "Particle " << pid << " failed: " << e.what() << "\n";
            }

            if ((pid + 1) % 50 == 0 || pid + 1 == cloud.size()) {
                std::cout << "Processed " << (pid + 1) << " / " << cloud.size() << " injection points\n";
            }
        }

        caustics_rz.close();
        caustics_equator.close();

        // Единственный файл поля концентрации.
        rz_grid.save_csv("stationary_rz_concentration_grid.csv", "r", "z");

        std::cout << "Done. Output files:\n";
        std::cout << "  stationary_rz_concentration_grid.csv\n";
        std::cout << "  caustic_points_rz.csv\n";
        std::cout << "  caustic_points_equator.csv\n";
        std::cout << "Accepted steps: " << total_accepted << "\n";
        std::cout << "Rejected steps: " << total_rejected << "\n";
        std::cout << "Completed trajectories: " << particles_completed << "\n";
        std::cout << "Trajectories hit core: " << particles_hit_core << "\n";
        std::cout << "Failed trajectories: " << particles_failed << "\n";
        std::cout << "Concentration samples used: " << total_samples << "\n";
        std::cout << "Caustics r-z: " << total_caustics_rz << "\n";
        std::cout << "Caustics equator: " << total_caustics_equator << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
