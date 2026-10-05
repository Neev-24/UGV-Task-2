#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using namespace std::chrono_literals;

class UgvController : public rclcpp::Node
{
public:
    UgvController() : Node("ugv_controller")
    {
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel",
            10,
            std::bind(
                &UgvController::cmdVelCallback,
                this,
                std::placeholders::_1
            )
        );

        joint_state_pub_ =
            this->create_publisher<sensor_msgs::msg::JointState>(
                "/joint_states",
                10
            );

        last_time_ = this->now();

        timer_ = this->create_wall_timer(
            20ms,
            std::bind(&UgvController::update, this)
        );

        RCLCPP_INFO(this->get_logger(), "UGV controller started");
    }

private:

    void cmdVelCallback(
        const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        constexpr double wheel_radius = 0.12;
        constexpr double wheel_separation = 0.58;

        double linear_velocity = msg->linear.x;
        double angular_velocity = msg->angular.z;

        left_velocity_ =
            (linear_velocity -
             angular_velocity * wheel_separation / 2.0)
            / wheel_radius;

        right_velocity_ =
            (linear_velocity +
             angular_velocity * wheel_separation / 2.0)
            / wheel_radius;
    }

    void update()
    {
        auto now = this->now();

        double dt = (now - last_time_).seconds();

        if (dt <= 0.0)
            return;

        left_position_ += left_velocity_ * dt;
        right_position_ += right_velocity_ * dt;

        sensor_msgs::msg::JointState joint_state;

        joint_state.header.stamp = now;

        joint_state.name = {
            "front_left_wheel_joint",
            "rear_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_right_wheel_joint"
        };

        joint_state.position = {
            left_position_,
            left_position_,
            right_position_,
            right_position_
        };

        joint_state.velocity = {
            left_velocity_,
            left_velocity_,
            right_velocity_,
            right_velocity_
        };

        joint_state_pub_->publish(joint_state);

        last_time_ = now;
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr
        cmd_vel_sub_;

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr
        joint_state_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    double left_velocity_ = 0.0;
    double right_velocity_ = 0.0;

    double left_position_ = 0.0;
    double right_position_ = 0.0;

    rclcpp::Time last_time_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<UgvController>()
    );

    rclcpp::shutdown();

    return 0;
}