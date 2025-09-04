#include "rm_utils/math/New_Extended_Kalman_Filter.hpp"

#include <numeric>




namespace fyt {

ExtendedKalmanFilter::ExtendedKalmanFilter(const Eigen::VectorXd &X0,const Eigen::MatrixXd &P,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> x_add):
        _X(X0),_P(P),_I(Eigen::MatrixXd::Identity(X0.rows(),X0.rows())),_x_add(x_add){
            _KF_data["residual_yaw"] = 0.0;     // 观测残差（偏航方向，Yaw）
            _KF_data["residual_pitch"] = 0.0;   // 观测残差（俯仰方向，Pitch）
            _KF_data["residual_distance"] = 0.0;// 观测残差（目标距离）
            _KF_data["residual_angle"] = 0.0;   // 观测残差（角度综合指标）
            _KF_data["nis"] = 0.0;              //NIS检验观测一致性
            _KF_data["nees"] = 0.0;             //NEES检验状态估计一致性
            _KF_data["nis_fail"] = 0.0;         //NIS失败次数
            _KF_data["nees_fail"] = 0.0;        //NEES失败次数
            _KF_data["recent_nis_failures"] = 0.0; // 最近一段时间内 NIS 检验失败的次数/比例（用于统计趋势）

}


Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd &F,const Eigen::MatrixXd & Q){

    return predict(F,Q,[&](const Eigen::VectorXd &x){return F*x;});

}

Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd &F,const Eigen::MatrixXd & Q,
                                              std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f){


    _X = f(_X);
    _P = F*_P*F.transpose() + Q;
    return _X;
}


Eigen::VectorXd ExtendedKalmanFilter::update(const Eigen::VectorXd &Z,const Eigen::MatrixXd &R,const Eigen::MatrixXd &H,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> z_subtract){
   
    return update(Z,R,H,[&](const Eigen::VectorXd &x){return H*x;},z_subtract);
}


Eigen::VectorXd ExtendedKalmanFilter::update(const Eigen::VectorXd &Z,const Eigen::MatrixXd &R,const Eigen::MatrixXd &H,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &)>h,
        std::function<Eigen::VectorXd(const Eigen::VectorXd &,const Eigen::VectorXd &)> z_subtract){

    Eigen::VectorXd x_prior = _X;
    Eigen::MatrixXd K = _P*H.transpose()*(H*_P*H.transpose()+ R).inverse();


    //更新_X
    _X = _x_add(_X,K*z_subtract(Z,h(_X)));
    
    //Joseph form（约瑟夫形式）的协方差更新公式
    _P = (_I-K*H)*_P*(_I-K*H).transpose()+K*R*K.transpose();
    

    //卡方检验
    Eigen::VectorXd residual = z_subtract(Z,h(_X));

    Eigen::MatrixXd S = H*_P*H.transpose()+R;
    double nis = residual.transpose()*S.inverse()*residual;       

    Eigen::VectorXd e_k = _X - x_prior;
    double nees = e_k.transpose()*_P.inverse()*e_k;

    //设定阈值
    constexpr double nis_thr = 0.711;
    constexpr double nees_thr = 0.711;

    if(nis>nis_thr) nis_count++,_KF_data["nis_fail"] = 1;
    if(nees>nees_thr) nees_count++,_KF_data["nees_fail"] = 1;

    total_count++;
    last_nis = nis;


    recent_nis_failures.push_back(nis>nis_thr?1:0);
    if(recent_nis_failures.size()>window_size){
        recent_nis_failures.pop_front();
    }

    int recent_failures = std::accumulate(recent_nis_failures.begin(),recent_nis_failures.end(),0);
    double recent_rate = static_cast<double>(recent_failures) / recent_nis_failures.size();

    _KF_data["residual_yaw"] = residual[0];
    _KF_data["residual_pitch"] = residual[1];
    _KF_data["residual_distance"] = residual[2];
    _KF_data["residual_angle"] = residual[3];
    _KF_data["nis"] = nis;
    _KF_data["nees"] = nees;
    _KF_data["recent_nis_failures"] = recent_rate;

    return _X;

}



}