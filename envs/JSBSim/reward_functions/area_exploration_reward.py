from scipy.spatial import ConvexHull
import numpy as np
from .reward_function_base import BaseRewardFunction  # 假设存在这个基类


class AreaExplorationReward(BaseRewardFunction):
    """
    AreaExplorationReward
    奖励策略：在指定范围内、指定时间窗口内最大化探索区域面积
    - 探索面积越大，奖励越高
    - 超出指定范围的探索不计入面积
    - 只考虑最近指定时间段内的探索
    """

    def __init__(self, config):
        super().__init__(config)
        # 从配置中获取参数，设置默认值
        self.exploration_radius = 500000.0 # 探索范围半径
        self.time_window = 30  # 时间窗口(秒)
        self.reward_scaling =  0.008  # 奖励缩放系数

    def get_reward(self, task, env, agent_id):
        """
        计算奖励：基于智能体在指定范围内、指定时间内探索的区域面积

        Args:
            task: 任务实例
            env: 环境实例
            agent_id: 智能体ID

        Returns:
            float: 奖励值
        """
        agent = env.agents[agent_id]
        current_time = env.current_step # 假设环境提供获取当前时间的方法

        # 1. 获取时间窗口内的位置历史
        # 假设智能体有position_history属性，存储格式为[(timestamp, x, y), ...]
        recent_positions = [
            (pos[1], pos[2])  # 提取x, y坐标
            for pos in agent.position_history
            if current_time - pos[0] <= self.time_window  # 筛选时间窗口内的位置
        ]

        if len(recent_positions) < 3:  # 至少需要3个点才能计算面积
            return self._process(0.0, agent_id)

        # 2. 过滤超出探索范围的位置
        # 假设以起始位置为探索中心
        start_x, start_y = agent.start_position  # 假设智能体有起始位置属性
        valid_positions = []
        for (x, y) in recent_positions:
            # 计算与起点的距离
            distance = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
            if distance <= self.exploration_radius:
                valid_positions.append((x, y))

        if len(valid_positions) < 3:
            return self._process(0.0, agent_id)

        # 3. 计算凸包面积（代表探索到的区域）
        try:
            hull = ConvexHull(valid_positions)
            exploration_area = hull.area
        except:
            # 处理共线点等无法计算凸包的情况
            exploration_area = 0.0

        # 4. 计算奖励（面积越大奖励越高）
        reward = exploration_area * self.reward_scaling
        # print(f"{agent_id}area_reward,{reward}")
        return self._process(reward, agent_id)
