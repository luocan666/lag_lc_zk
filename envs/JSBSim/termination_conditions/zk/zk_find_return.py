from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition


class ZKFindReturn(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        # print(f'detected enemy_plane, {env.agents[agent_id].single_detected_enemies}')
        if env.agents[agent_id].single_detected_enemies != []:
            self.log(f"{agent_id} detected enemy! Total Steps={env.current_step}")
            return True,False,info
        else:
            return False, False, info
