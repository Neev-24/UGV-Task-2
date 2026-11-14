#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include <gazebo/common/MouseEvent.hh>
#include <gazebo/gui/GuiIface.hh>
#include <gazebo/gui/GuiPlugin.hh>
#include <gazebo/gui/MouseEventHandler.hh>
#include <gazebo/rendering/UserCamera.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

class UGVClickGoalGui : public gazebo::GUIPlugin
{
  Q_OBJECT

public:
  UGVClickGoalGui() : gazebo::GUIPlugin()
  {
    // This plugin listens to the 3-D view without adding an overlay widget.
    this->move(0, 0);
    this->resize(0, 0);
  }

  ~UGVClickGoalGui() override
  {
    if (registered_) {
      gazebo::gui::MouseEventHandler::Instance()->RemoveReleaseFilter(filter_name_);
    }
  }

  void Load(sdf::ElementPtr /*sdf*/) override
  {
    node_ = gazebo_ros::Node::CreateWithArgs("ugv_click_goal_gui");
    publisher_ = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "/ugv/click_goal", rclcpp::QoS(10));
    gazebo::gui::MouseEventHandler::Instance()->AddReleaseFilter(
      filter_name_, std::bind(&UGVClickGoalGui::OnMouseRelease, this, std::placeholders::_1));
    registered_ = true;
    RCLCPP_INFO(node_->get_logger(), "Click-goal GUI ready: left-click the arena floor.");
  }

private:
  bool OnMouseRelease(const gazebo::common::MouseEvent & event)
  {
    if (event.Button() != gazebo::common::MouseEvent::LEFT || event.Dragging()) {
      return false;
    }
    auto camera = gazebo::gui::get_active_camera();
    if (!camera || !publisher_) {
      return false;
    }

    ignition::math::Vector3d origin, direction;
    camera->CameraToViewportRay(event.Pos().X(), event.Pos().Y(), origin, direction);
    if (std::abs(direction.Z()) < 1e-9) {
      return false;
    }
    const double t = -origin.Z() / direction.Z();
    if (!std::isfinite(t) || t <= 0.0) {
      return false;
    }
    const auto point = origin + direction * t;
    if (!std::isfinite(point.X()) || !std::isfinite(point.Y())) {
      return false;
    }

    geometry_msgs::msg::PointStamped goal;
    goal.header.frame_id = "world";
    goal.header.stamp = node_->now();
    goal.point.x = point.X();
    goal.point.y = point.Y();
    goal.point.z = 0.0;
    publisher_->publish(goal);
    // Let Gazebo's normal selection and camera controls receive the event too.
    return false;
  }

  const std::string filter_name_ = "ugv_click_goal_gui";
  bool registered_ = false;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr publisher_;
};

GZ_REGISTER_GUI_PLUGIN(UGVClickGoalGui)

#include "click_goal_gui.moc"
