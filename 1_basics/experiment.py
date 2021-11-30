def episode(agent, render_episode, params, nr_episode=0):
    env = params["env"]
    state = env.reset()
    discounted_return = 0
    gamma = params["gamma"]
    done = False
    time_step = 0
    while not done:
        if render_episode:
            env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += reward*(gamma**time_step)
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return