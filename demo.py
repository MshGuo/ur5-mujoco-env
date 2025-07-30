import mujoco
import time
import glfw
import numpy as np
import cv2
from pick_box_env import PickBoxEnv

if __name__=="__main__":
    env = PickBoxEnv()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        env.reset()
        start_pos, start_rotm = env.get_current_pose()

        done = False
        cnt = -1
        step = -1
        step_res = True


        while viewer.is_running():
            cnt += 1
            step_start = time.time()

            if step == -1 and step_res:
                step = 0
                step_res = False
                env.need_plan = True
                start_pos, start_rotm = env.get_current_pose()
                box_pos, box_eulerz = env.get_box_pos(env.pick_box)
                target = [0.0, 0.0, 0.0, np.pi, 0.0, np.pi / 2]
                target[:3] = box_pos + [0.0, 0.0, 0.3]
                start_time = env.data.time

            elif step == 0 and step_res:  # 步骤0完成，准备步骤1参数，下移
                step = 1
                step_res = False
                env.need_plan = True
                target[:3] = box_pos + [0.0, 0.0, 0.17]
                target[3:] = [np.pi, 0.0, np.pi / 2 + box_eulerz]
                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time

            elif step == 1 and step_res:  # 步骤2，抓取上移
                env.gripper_close()
                step = 2
                step_res = False
                env.need_plan = True
                target[:3] = box_pos + [0.0, 0.0, 0.3]
                target[3:] = [np.pi, 0.0, np.pi / 2]

                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time

            elif step == 2 and step_res:  # 步骤3，移动到目标物上方
                step_res = False
                step = 3
                env.need_plan = True
                box_pos, box_eulerz = env.get_box_pos(env.place_box)

                target[:3] = box_pos + [0.0, 0.0, 0.3]

                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time

            elif step == 3 and step_res:  # 步骤4，移动到目标物处
                step_res = False
                step = 4
                env.need_plan = True
                target[:3] = box_pos + [0.0, 0.0, 0.20]
                target[3:] = [np.pi, 0.0, np.pi / 2 + box_eulerz]
                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time
            elif step == 4 and step_res:
                env.gripper_open()
                done = True
            else:
                pass

            step_res, action_pos, action_euler = env.line_move(start_pos, start_rotm, target[:3], target[3:],
                                                               start_time,
                                                               env.data.time)
            if cnt % 20 == 0:
                env.cam_tip.show_img()
                env.cam_world.show_img()

            mujoco.mj_step(env.model, env.data)

            viewer.sync()
            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        glfw.terminate()
        cv2.destroyAllWindows()