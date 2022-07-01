#include <complex>
#include <iostream>
#include <fmt/format.h>
#include <Eigen/Dense>

#include <ranges>
#include <math.h>
namespace ei = Eigen;


void diffurs();
int main() {
    diffurs();
    return 0;
}












void grids();
void tsk_X_8_1();
void tsk_XI_91();
void tsk_XI_93();


constexpr int tasks_end=16;
volatile int do_task_n=0;

void diffurs()
{
       while (do_task_n != tasks_end)
    {
        switch (do_task_n)
        {
        case 0:
            tsk_X_8_1();
            break;
        case 1:
            tsk_XI_91();
            break;
        case 2:
            tsk_XI_93();
            break;
        case 3:
            grids();
            break;
        default:
            break;
        }
        auto n=do_task_n;
        do_task_n=n+1;
    }
}

template <class F, class X>
double diff(F f, X x, double h = 0.0001)
{
    return (f(x + h) - f(x - h)) / (2 * h);
}

template <class F, class X>
auto find_root_newton(F f, X x0, double y = 0.0, double err = 0.0001)
{
    // f(x)-y=0
    auto fun = [&](double _x)
    { return f(_x) - y; };
    decltype(f(x0)) xn = x0;
    while (err < std::abs(fun(xn)))
    {
        xn = xn - fun(xn) / diff(fun, xn, err / 10);
    }
    return xn;
}

template <class F>
auto applyN(std::vector<F> funcs, ei::VectorXd x)
{
    int N = funcs.size();
    ei::VectorXd fy(N);
    for (int i : std::views::iota(0, N))
    {
        fy(i) = funcs[i](x);
    }
    return fy;
};

template <class F>
class jacobian
{
private:
    F &functor;
    int N, M;

public:
    //
    // f : Rn → Rm
    // f_i(VectorXd)
    //
    //
    jacobian(F &f, int _M, int _N)
        : functor(f), N(_N), M(_M)
    {
    }

    ei::MatrixXd compute(ei::VectorXd &x)
    {
        // N=x.size();
        ei::MatrixXd J(M, N);
        for (int m : std::views::iota(0, M))
        {
            for (int n : std::views::iota(0, N))
            {
                auto y_i = [&](auto dx)
                { 
                    ei::VectorXd xx{x};
                    xx[n]+=dx;
                    return functor(xx)[m]; };
                J(m, n) = diff(y_i, 0);
            }
        }
        return J;
    }
};

template <class F,size_t iter_limit=10000ul>
class find_root_N
{
private:
    // using R=std::invoke_result_t<F>;
    F &functor;
    jacobian<F> J;
    double error(ei::VectorXd x)
    {
        return functor(x).norm();
    }

public:
    find_root_N(F &f, int m, int n)
        : functor(f),J(f, m, n)  {}

    auto compute(ei::VectorXd &x0, double err = 0.0001)
    {
        auto xn = x0;
        auto iters=iter_limit;
        while ( err < error(xn) and --iters>0)
        {
            auto j = J.compute(xn);
            ei::VectorXd dx = j.colPivHouseholderQr().solve(-1 * functor(xn));
            xn = xn + dx;
        }
        if (iters==0)
        {
            fmt::print("Ireartion limit reached, using current guess");
        }
        
        return xn;
    }
};

template <class... Fs>
auto create_vector_functor(Fs... f)
{
    std::vector<std::common_type_t<Fs...>> funcs;
    (funcs.push_back(f), ...);
    auto ret_lambda= [=](double x, ei::VectorXd y) -> ei::VectorXd
    {
        int N = sizeof...(f);
        ei::VectorXd fy(N);
        for (int i : std::views::iota(0, N))
        {
            fy(i) = funcs[i](x, y);
        }
        return fy;
    };
    return ret_lambda;
}

template <int N>
struct ruunge_params
{
    ei::Vector<double, N> c, b;
    ei::Matrix<double, N, N> A;
};

struct eig_shape
{
    int rows;
    int cols;
};
template <typename Derived>
constexpr eig_shape print_size(const Eigen::EigenBase<Derived> &b)
{
    return {b.rows(), b.cols()};
}
namespace RK
{
template <int S, class F>
class ruunge
{   
    F &functor;
    ei::Vector<double, S> c, b;
    ei::Matrix<double, S, S> A;
    ei::VectorXd y_n;
    double x_n;
    double h;
    double error;
    int M;

public:
    using functor_type_example_t =ei::VectorXd (*)(double ,ei::VectorXd);
    // y'=f(x,y) , y<--R^M
    // k_i=f(x_n + c_i*h , y_n + h*sum_j(a_ij*k_j))   --> system of eq  g_i(k1,...,k_s)= k_i-f(x_n + c_i*h_i , y_n + h*sum_j(a_ij*k_j))=0
    // k_i: R^m ==> g_i: R^(m*N)-->R^m
    // y_(n+1)= y_n + h*sum_i(b_i*k_i)
    ruunge(ruunge_params<S> params, F &f)
        : functor(f), c(params.c), b(params.b), A(params.A)
    { }

    //   | k_11  ....  k_M1
    // K=| .
    //   | k_1S
    auto solve_for_k()
    {

        auto dk = [&](ei::VectorXd &ks, int i) -> ei::VectorXd
        {
            ei::VectorXd out(M);
            for (int j = 0; j < S; j++)
            {
                out += A(i, j) * (ks(ei::seqN(j * M, M)));
            }
            return out;
        };
        auto g_i = [&](ei::VectorXd &ks, int i) -> ei::VectorXd
        { return ks(ei::seqN(i * M, M)) - functor(x_n + h * c(i), y_n + h * (dk(ks, i))); };

        auto G = [&](ei::VectorXd &ks)
        {
            ei::VectorXd res(M * S);
            for (auto &&i : std::views::iota(0, S))
            {
                res(ei::seqN(i * M, M)) = g_i(ks, i);
            }
            return res;
        };
        ei::VectorXd keys(S * M);
        find_root_N root(G, M * S, M * S);
        ei::VectorXd ret = root.compute(keys,error);
        return ret.reshaped(M, S).eval();
    }
    auto step()
    {
        ei::MatrixXd K = solve_for_k();
        ei::VectorXd dy = ei::VectorXd::Zero(M);
        for (auto i : std::views::iota(0, S))
        {
            auto k_i = K.col(i);
            dy += b(i) * k_i;
        }
        y_n += h * dy;
        x_n += h;
        return y_n;
    }


    auto compute(ei::VectorXd y0,double x_0,double x_end, int steps,double err=0.0001){
        h=(x_end-x_0)/steps;
        M=y0.size();
        y_n=y0;
        x_n=x_0;
        error=err;
        
        for (auto &&j : std::views::iota(0, steps))
        {
            (void)j;
            step();
        }

    }
};



   ruunge_params<4> four_stage{
        {1.0 / 2, 2.0 / 3, 1.0 / 2, 1.0},
        {3.0 / 2, -3.0 / 2, 1.0 / 2, 1.0},
        ei::Matrix<double, 4, 4>{
            { 1.0 / 2,         0,       0,   0},
            { 1.0 / 6,   1.0 / 2,       0,   0},
            {-1.0 / 2,   1.0 / 2, 1.0 / 2,   0},
            { 3.0 / 2,  -3.0 / 2, 1.0 / 2, 1.0 / 2}}};
    ruunge_params<1> euler_backward{
        ei::Vector<double, 1>{1},
        ei::Vector<double, 1>{1},
        ei::Matrix<double, 1, 1>{1}};
    ruunge_params<2> second_backward{
        ei::Vector<double, 2>{{0, 1}},
        ei::Vector<double, 2>{{1.0 / 2, 1.0 / 2}},
        ei::Matrix<double, 2, 2>{{      0,          0},
                                 {1.0 / 2,    1.0 / 2}}};

} // namespace hm

void tsk_X_8_1()
{
    //  y″ – 10 y′ – 11 y = 0, y(0) = 1, y′(0) = –1, t=0..10
    //   __
    //  | y'=u           y(0) = 1,      t=0..10   |y|'  |0  1 ||y|
    //  | u'=11y+10u     u(0) = –1                | | = |     || |
    //  |__                                       |u|   |11 10||u|
    Eigen::VectorXd b{{0, 0}};
    Eigen::VectorXd y_0{{1, -1}};
    Eigen::MatrixXd A{{0, 1},
                      {11, 10}};

    auto functor = [=]([[maybe_unused]] double x,ei::VectorXd y) 
    { return ei::VectorXd{A * y}; };
    RK::ruunge i(RK::four_stage, functor);
    i.compute(y_0, 0.0,4.0,100,0.0001);




}

void tsk_XI_91()
{

    /* y''-(10+x^2)*y=x*exp(-x) ==> y'= 0*y +   u
                                    u'=(10+x^2)y+0*u+x*exp(-x)
    */
    auto F = [](double x, ei::VectorXd y)
                                        { ei::VectorXd ret(2);
                                        ret(0)=y(1);
                                        ret(1)=-(10+x*x)*y(0)+x*std::exp(-x);
                                        return ret; };
    ei::VectorXd y0{{1, 0.1}};

    RK::ruunge I(RK::second_backward, F);
    I.compute(y0,0,1,10,0.0001);

}

void tsk_XI_93()
{
    /* y''-x*sqrt(y)=0 ==>          y'= 0*y +   u      x=0...1
                                    u'=x*sqrt(y)+0*u   y(0)=0 ,y(1)=2
    */
    auto F = [](double x, ei::VectorXd y)
    { ei::VectorXd ret(2);
                                        ret(0)=y(1);
                                        double yy=y(0);
                                        double d=x*std::sqrt(yy);
                                        ret(1)=d;
                                        return ret; };


    for (size_t param = 0; param < 20; param++)
    {
        /* code */

        ei::VectorXd y0{{0.0001, param / 8.0}};
        RK::ruunge I(RK::second_backward, F);
        I.compute(y0,0,1,10,0.001);
        std::vector<double> gy;
        gy.push_back(y0(0));
        for ( [[maybe_unused]] auto &&j : std::views::iota(0, 20))
        {
           
            gy.push_back(I.step()(0));
        }
        auto y = I.step()(0);
       // fmt::print("{}\n", y);
    }
}


template <class Func>
class GridSolver
{
public:
    ei::MatrixXd G;
    double tau, h;
    double x0, t0;
    double k;
    //  double (&F)(double, double, double);
    Func &F;
    using Func_t = decltype(F);
    int P;
    int M;
    GridSolver(int _P, int _M, ei::VectorXd top, ei::VectorXd left, ei::VectorXd right, double _k, Func &_F,
               double _x0, double _xM, double _t0, double _tP)
        : x0(_x0), t0(_t0),  k(_k), F(_F), P(_P), M(_M)
    {
        G = ei::MatrixXd::Zero(P, M);
        tau = (_tP - t0) / P;
        h = (_xM - x0) / M;
        G(ei::all, 0) = left;
        G(ei::all, ei::last) = right;
        G(0, ei::all) = top;
        // std::cout<<G;
    }

    double x(int m) { return x0 + h * m; }
    double dx(int m, int p) { return (G(p, m + 1) - G(p, m - 1)) / (2 * h); }
    double t(int p) { return t0 + tau * p; }

    void move_up()
    {

        auto movement = [&](int m, int p, double y)
        {
            return (y - G(p, m)) / tau -
                   k * k * (G(p, m - 1) - 2 * y + G(p, m + 1)) / (h * h) -
                   F(x(m), t(p), dx(m, p));
        };
        // fmt::print("G has {} rows and {}cols \n",G.rows(),G.cols());
        for (auto &&p : std::views::iota(0, P - 1))
        {
            for (auto &&m : std::views::iota(1, M - 1))
            {
                double next = find_root_newton([&](double y)
                                               { return movement(m, p, y); },
                                               G(p, m) + h);
                G(p + 1, m) = next;
            }
            std::cout << G << "\n\n";
        }
    }
};

void grids()
{

    auto f = [](double x, double t, double ddx)
    { return ddx * (4 * x / (x * x + 2 * t + 1)); };
    auto u_0_t = [](double t)
    { return (1.0 / (2 * t + 1)); };
    auto u_1_t = [](double t)
    { return (0.5 / (t + 1)); };
    auto u_x_0 = [](double x)
    { return (1.0 / (x * x + 1)); };

    int M = 10;
    int P = 10;

    double x0 = 0;
    double t0 = 0;

    double xM = 1;
    double tP = 1;

    ei::VectorXd left(P);
    ei::VectorXd right(P);
    ei::VectorXd top(M);
    for (auto &&i : std::views::iota(0, P))
    {
        left(i) = u_0_t(std::lerp(t0, tP, 1.0 * i / P));
        right(i) = u_1_t(std::lerp(t0, tP, 1.0 * i / P));
    }
    for (auto &&j : std::views::iota(0, M))
    {
        top(j) = u_x_0(std::lerp(x0, xM, 1.0 * j / M));
    }

    GridSolver g(P, M, top, left, right, 1, f, x0, xM, t0, tP);
    g.move_up();
}







