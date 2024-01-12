import time

from dense_tag import dense_tag_env

env = dense_tag_env(render_mode=None, num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True,)
# env = aec_to_parallel(env)

i = 0
env.reset()
total_reward = {agent: 0 for agent in env.agents}
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(f"{i}: Agent: {agent}, Truncation: {truncation}, Info: {info}")
    i += 1
    total_reward[agent] += reward


    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy

    env.step(action)
    # time.sleep(0.1)
    # print(agent, reward)
    if termination:
        print('terminated')
    if truncation:
        print('truncated ', i)
    # if termination or truncation:
    #     env.reset()
    #     i = 0
env.close()
print("Total Reward: ", total_reward)