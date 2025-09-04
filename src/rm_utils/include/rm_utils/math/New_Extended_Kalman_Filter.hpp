#ifndef RM_UTILS_NEW_EXTENDED_KALMAN_FILTER_HPP_
#define RM_UTILS_NEW_EXTENDED_KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>



namespace fyt{

class ExtendedKalmanFilter{

public:
    Eigen::VectorXd _X; // 状态
    Eigen::MatrixXd _P; //协方差
    std::map<std::string,double> _KF_data; //卡方检验的数据

    std::deque<int> recent_nis_failures{0};//用来储存这段时间的NIS是否失败
    size_t window_size = 100;//储存NIS是否失败的最大值
    double last_nis;//上一次更新的NIS值

    ExtendedKalmanFilter() = default;

    ExtendedKalmanFilter(const Eigen::VectorXd &X0,const Eigen::MatrixXd & P,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> x_add=
        [](const Eigen::VectorXd &a,const Eigen::VectorXd &b){return a+b;});
    

    Eigen::VectorXd predict(const Eigen::MatrixXd & F,const Eigen::MatrixXd &Q);//退化成KF的线性卡尔曼

    Eigen::VectorXd predict(const Eigen::MatrixXd & F,const Eigen::MatrixXd &Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);//EKF的形式


    Eigen::VectorXd update(const Eigen::VectorXd &Z,const Eigen::MatrixXd &R,const Eigen::MatrixXd &H,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> z_subtract=
        [](const Eigen::VectorXd &a,const Eigen::VectorXd &b){return a-b;});

    Eigen::VectorXd update(const Eigen::VectorXd &Z,const Eigen::MatrixXd &R,const Eigen::MatrixXd &H,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)>h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> z_subtract=
        [](const Eigen::VectorXd &a,const Eigen::VectorXd &b){return a-b;});






private:
    Eigen::MatrixXd _I;//单位矩阵
    std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> _x_add;//处理状态相加的函数


    int nis_count = 0;//代码运行过程中所有NIS验证失败的次数
    int nees_count = 0;//代码运行过程中所有NEES验证失败的次数
    int total_count = 0;//更新次数






};





}





#endif