#include<iostream>
#include<ctime>
#include<Eigen/Core>
#include<Eigen/Dense>

#define MATRIX_SIZE 50

using namespace std;

int main()
{
    //1、矩阵声明
    Eigen::Matrix<float, 2, 3> matrix_23;
    //3d向量和矩阵封装类，本质上还是Eigen::Matrix<double, x, x>
    Eigen::Vector3d v_3d;
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();

    //动态矩阵声明
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;


    matrix_23 << 1,2,3,4,5,6;
    cout << matrix_23 << endl;

    for(int i=0;i<1;i++)
        for(int j=0;j<2;j++)
            cout << matrix_23(i,j) << endl;
    
    //2、矩阵运算
    //不能混合两种不同类型的矩阵，float和double不能相互混合
    v_3d << 3,2,1;
    //使用cast<>()可以强制转换矩阵元素类型
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << result << endl;

    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl<<endl;
    cout << matrix_33.transpose() << endl;    //转置
    cout << matrix_33.sum() << endl; //各元素和
    cout << matrix_33.trace() << endl;  //迹
    cout << 10*matrix_33 << endl;  //数乘
    cout << matrix_33.inverse() << endl;  //逆
    cout << matrix_33.determinant() << endl;  //行列式

    //特征值，与自身转置相乘
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver (matrix_33.transpose()*matrix_33);
    cout << "Eigen values = " << eigen_solver.eigenvalues() << endl;
    cout << "Eigen values = " << eigen_solver.eigenvectors() << endl;

    //解方程 matrix_NN * x = v_Nd
    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);
    clock_t time_stt = clock();     //计时
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse()*v_Nd;
    cout << "time use in normal inverse is " << 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC << 'ms' << endl;
    //另一种求解方法，矩阵分解求解例如QR分解，速度快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "tim use Qr compsition is " << 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC << 'ms' << endl;


    return 0;
}