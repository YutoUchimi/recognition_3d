#!/usr/bin/env roseus

(ros::load-ros-manifest "jsk_gui_msgs")
(ros::roseus "save_image")

(defun save-image ()
  (let (can-save)
    (ros::ros-info "start save-image.l")
    (ros::wait-for-service "/rviz/yes_no_button")
    (while t
      (while
          (not can-save)
        (setq can-save (send (ros::service-call
                              "/rviz/yes_no_button" (instance jsk_gui_msgs::YesNoRequest)) :yes)))

      (setq res
            (ros::service-call "/data_collection_server/save_request"
                               (instance std_srvs::TriggerRequest :init)))
      (ros::ros-info "success: ~A, message: ~A"
                           (send res :success) (send res :message))

      (setq can-save nil)
      )))


(defun main ()
  (save-image)
  )

(main)
