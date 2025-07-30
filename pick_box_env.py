import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import glfw
import os
os.environ['MUJOCO_GL'] = 'glfw'


class env_cam():
    def __init__(self, model, data, camera_name, width=640, height=480):
        self.model = model
        self.data = data
        self.name = camera_name

        self.width, self.height = width, height
        self.rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # self.depth_buffer = np.zeros((self.height, self.width), dtype=np.float32)

        # 初始化GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        # 创建窗口
        window = glfw.create_window(self.width, self.height, camera_name, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        glfw.make_context_current(window)

        # 设置相机视角
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera.fixedcamid = camera_id

        # 创建渲染上下文和场景
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        # 渲染选项
        self.vopt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()

    def show_img(self, show=True):
        # 更新场景
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.perturb, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)

        # 渲染相机视图
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scene, self.context)

        # 读取RGB图像
        mujoco.mjr_readPixels(self.rgb_buffer, None, viewport, self.context)

        # 由于MuJoCo返回的图像是上下颠倒的，需要翻转它
        rgb_image = np.flipud(self.rgb_buffer)

        if show:
            # 转换为BGR格式供OpenCV使用
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, bgr_image)
            cv2.waitKey(1)

        # 处理GLFW事件
        glfw.poll_events()

class PickBoxEnv():
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('./ur5e_robotiq2f85/scene.xml')
        self.data = mujoco.MjData(self.model)

        joint_names=["shoulder_pan_joint",
                     "shoulder_lift_joint",
                     "elbow_joint",
                     "wrist_1_joint",
                     "wrist_2_joint",
                     "wrist_3_joint"]
        self.joint_ids = [self.model.joint(name).id for name in joint_names]
        self.actuator_ids = [self.model.actuator(i).id for i in range(self.model.nu)]
        self.actuator_ids = self.actuator_ids[:6]
        self.gripper_id = self.model.actuator("fingers_actuator").id
        self.site_id = self.model.site("attachment_site").id
        # self.key_id = self.model.key("home").id
        self.box1_id = self.model.body("box1").id
        self.box2_id = self.model.body("box2").id
        self.box3_id = self.model.body("box3").id

        # 逆运动学相关参数
        self.MAX_JOINT_STEP = 0.05
        self.TOLERANCE = 0.002
        self.integration_dt: float = 1.0
        damping: float = 1e-4
        self.dt = 0.002

        self.jac = np.zeros([6, self.model.nv])
        self.diag = damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]

        self.cur_pos = np.zeros(3)
        self.cur_rotm = np.zeros([3,3])
        self.cur_quat = np.zeros(4)
        self.cur_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        self.move2init()

        # 笛卡尔直线运动规划
        self.v = 0.05
        self.a = 0.05
        self.t1, self.t2, self.t3 = 0, 0, 0
        self.d1, self.d2, self.d3 = 0, 0, 0
        self.dx, self.dy, self.dz, self.d = 0, 0, 0, 0

        # 随机种子
        self.seed = None
        self.set_seed()

        self.cam_world = env_cam(self.model, self.data, 'cam_world')
        self.cam_tip = env_cam(self.model, self.data, 'cam_tip')

    def reset(self):
        self.move2init()
        self.gripper_open()
        self.set_box_pos_random()
        self.choose_box()

    def set_seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def line_plan(self, start_pos, start_rotm, end_pos, end_euler):
        v = self.v
        a = self.a

        self.dx = end_pos[0] - start_pos[0]
        self.dy = end_pos[1] - start_pos[1]
        self.dz = end_pos[2] - start_pos[2]
        self.d = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        d = self.d
        if d > v**2 / a: # 有匀速段
            t1 =  v / a
            d1 = 0.5 * a * t1**2

            d2 = d - d1
            t2 = t1 + (d2 - d1) / v

            d3 = d
            t3 = t2 + t1
        else: # 无匀速段
            d1 = d / 2.0
            t1 = np.sqrt(d / a)

            t2 = t1
            d2 = d1

            t3 = 2.0 * t1
            d3 = d
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.need_plan = False

    def line_move(self, start_pos, start_rotm, end_pos, end_euler, start_time, cur_time):
        if self.need_plan:
            self.line_plan(start_pos, start_rotm, end_pos, end_euler)
        t = cur_time - start_time
        if t < self.t1:
            deltaD = 0.5 * self.a * t**2
        elif t < self.t2:
            deltaD = self.d1 + self.v*(t-self.t1)
        elif t <= self.t3:
            deltaT = t - self.t2
            deltaD = self.d2 + self.a*deltaT*(self.t1-0.5*deltaT)
        else:
            return True, end_pos, end_euler
        x = start_pos[0] + self.dx * deltaD / self.d
        y = start_pos[1] + self.dy * deltaD / self.d
        z = start_pos[2] + self.dz * deltaD / self.d
        self.move2pose(np.array([x, y, z]), end_euler, False)
        return False, np.array([x, y, z]), end_euler

    def get_current_pose(self):
        pos = self.data.site(self.site_id).xpos.copy()
        rotm = self.data.site(self.site_id).xmat.copy()
        return pos, rotm

    def move2pose(self, target_pos, target_rotm, flag):

        self.cur_pos, self.cur_rotm = self.get_current_pose()

        self.error_pos[:] = target_pos - self.cur_pos

        target_quat = np.zeros((4))
        if flag: # True 输入姿态为ROT
            mujoco.mju_mat2Quat(target_quat, target_rotm)
        else: # False 输入姿态为euler
            if len(target_rotm)!=3:
                print(11)
            mujoco.mju_euler2Quat(target_quat, target_rotm, 'XYZ')

        mujoco.mju_mat2Quat(self.cur_quat, self.cur_rotm)
        mujoco.mju_negQuat(self.cur_quat_conj, self.cur_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.cur_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        if np.linalg.norm(self.error[:3]) < self.TOLERANCE:
            return True

        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)
        # 限制最大步长
        dq = np.clip(dq, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)

        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)

        # Set the control signal.
        np.clip(q[:6], *self.model.jnt_range[:6].T, out=q[:6])
        self.data.ctrl[self.actuator_ids] = q[self.joint_ids]
        # mujoco.mj_step(self.model, self.data)
        return False

    def gripper_open(self):
        self.data.ctrl[self.gripper_id] = 0
        # mujoco.mj_step(self.model, self.data)

    def gripper_close(self):
        self.data.ctrl[self.gripper_id] = 255
        # mujoco.mj_step(self.model, self.data)

    def move2init(self):
        # mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        self.data.qpos[:6]=[0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        mujoco.mj_forward(self.model, self.data)

    def set_box_pos_random(self):
        # 随机范围：x[-0.7,-0.4], y[-0.2, 0.2], z=0.03, 绕z轴[-45,45]
        self.data.qpos[self.model.body_jntadr[self.box1_id]:self.model.body_jntadr[self.box1_id] + 3] = [self.rng.uniform(-0.7, -0.4), self.rng.uniform(-0.2, 0.2), 0.03]
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 7:self.model.body_jntadr[self.box1_id] + 10] = [self.rng.uniform(-0.7, -0.4), self.rng.uniform(-0.2, 0.2), 0.03]
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 14:self.model.body_jntadr[self.box1_id] + 17] = [self.rng.uniform(-0.7, -0.4), self.rng.uniform(-0.2, 0.2), 0.03]
        quat = np.zeros(4)
        z1 = self.rng.uniform(-np.pi / 4, np.pi / 4)
        z2 = self.rng.uniform(-np.pi / 4, np.pi / 4)
        z3 = self.rng.uniform(-np.pi / 4, np.pi / 4)

        mujoco.mju_euler2Quat(quat, [.0, .0, z1], 'xyz')
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 3:self.model.body_jntadr[self.box1_id] + 7] = quat

        mujoco.mju_euler2Quat(quat, [.0, .0, z2], 'xyz')
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 10:self.model.body_jntadr[self.box1_id] + 14] = quat

        mujoco.mju_euler2Quat(quat, [.0, .0, z3], 'xyz')
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 17:self.model.body_jntadr[self.box1_id] + 21] = quat

        mujoco.mj_forward(self.model, self.data)
        return np.array([z1, z2, z3])

    def get_box_pos(self, color):
        box_id = -1
        if color == 0:  # red
            box_id = self.box1_id
        elif color == 1:  # green
            box_id = self.box2_id
        else:  # blue
            box_id = self.box3_id
        pos = self.data.body(box_id).xpos.copy()
        rotm = self.data.body(box_id).xmat.copy()
        euler = self.rotm2rpy(rotm.reshape(3, 3))
        eulerz = euler[2]
        return pos, eulerz

    def choose_box(self):
        arr = np.array([0, 1, 2])
        self.pick_box, self.place_box = self.rng.choice(arr, size=2, replace=False)

    def rotm2rpy(self, R):
        # # XYZ顺序
        # pitch = np.arcsin(R[0, 2])
        # if np.isclose(abs(pitch), np.pi / 2):
        #     yaw = 0
        #     roll = np.arctan2(-R[1, 0], R[1, 1])
        # else:
        #     yaw = np.arctan2(-R[0, 1], R[0, 0])
        #     roll = np.arctan2(-R[1, 2], R[2, 2])
        # return np.array([roll, pitch, yaw])

        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if np.isclose(abs(pitch), np.pi / 2):
            yaw = 0
            roll = np.arctan2(R[0, 1], R[1, 1])
        else:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw])

