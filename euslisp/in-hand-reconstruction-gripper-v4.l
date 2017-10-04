#!/usr/bin/env roseus

(ros::roseus "robot_main")

(require "package://jsk_2016_01_baxter_apc/euslisp/lib/baxter-interface.l")
(require "package://jsk_2015_05_baxter_apc/euslisp/lib/util.l")
(require "package://jsk_2016_01_baxter_apc/euslisp/lib/util.l")

(defun init (&optional (ctype :default-controller) (calib-pressure nil))
  (jsk_2016_01_baxter_apc::baxter-init :ctype ctype)
  (send *ri* :gripper-servo-off)
  (send *baxter* :fold-pose-back)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  (if calib-pressure
      (send *ri* :calib-pressure-threshold)
    nil)
  (setq box (make-cube 100 100 100))
  (send box :set-color :blue)
  (send box :locate #f(800 0 0))
  (setq view-objects (list *baxter* box))
  (objects view-objects)
  (send *irtviewer* :change-background #f(0.5 0.5 0.5))
  (send *irtviewer* :redraw)
  )


(defun manipulate ()
  ;; hold an item with right hand
  (setq rarm-target-coords (send box :copy-worldcoords))
  (send rarm-target-coords :rotate (deg2rad 90) :z)
  (send rarm-target-coords :translate #f(0 0 100))
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (setq view-objects (append view-objects (list rarm-target-coords)))
  (objects view-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (send *ri* :start-grasp)
  (send *ri* :wait-interpolation)
  (send (send *baxter* :rarm :end-coords) :assoc box)

  ;; look at the item with left hand
  (setq larm-target-coords (send box :copy-worldcoords))
  (send larm-target-coords :translate #f(-200 400 0))
  (send larm-target-coords :rotate (deg2rad -60) :z)
  (send larm-target-coords :rotate (deg2rad -90) :y)
  (send *baxter* :larm :inverse-kinematics larm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (setq view-objects (append view-objects (list larm-target-coords)))
  (objects view-objects)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 5000)
  (send *ri* :wait-interpolation)

  ;; move the item with right hand
  ;; view 1
  (send rarm-target-coords :translate #f(0 0 50))
  (send rarm-target-coords :rotate (deg2rad -45) :y)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "1")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 2
  (send rarm-target-coords :translate #f(0 0 50) :world)
  (send rarm-target-coords :rotate (deg2rad -45) :y)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "2")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 3
  (send rarm-target-coords :rotate (deg2rad -45) :z :world)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "3")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 4
  (send rarm-target-coords :rotate (deg2rad -45) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "4")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 5
  (send rarm-target-coords :rotate (deg2rad -45) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "5")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 6
  (send rarm-target-coords :rotate (deg2rad -45) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "6")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 7
  (send rarm-target-coords :rotate (deg2rad -45) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "7")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; the same view as view 3
  (send rarm-target-coords :rotate (deg2rad 180) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)

  ;; view 8
  (send rarm-target-coords :rotate (deg2rad 45) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "8")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))

  ;; view 9
  (send rarm-target-coords :rotate (deg2rad 45) :z)
  (send *baxter* :rarm :inverse-kinematics rarm-target-coords
        :rotation-axis t
        :use-gripper t
        :move-palm-end t)
  (send *irtviewer* :redraw)
  (send *ri* :angle-vector (send *baxter* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (setq view_frame "9")
  (ros::set-param "view_frame" view_frame)
  (unix::sleep 3)
  ;; (setq res
  ;;       (ros::service-call "/data_collection_server/save_request"
  ;;                          (instance std_srvs::TriggerRequest :init)))
  ;; (ros::ros-info-green "view_frame: ~A, success: ~A, message: ~A"
  ;;                      view_frame (send res :success) (send res :message))
  )


(defun main ()
  (init)
  ;; (manipulate)
  )

(main)
