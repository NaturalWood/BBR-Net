from ppo import *
import matplotlib.pyplot as plt
import seaborn as sns
"""Need import yolo part"""
# from yolo import *
"""Need import yolo part"""
# import New_Pong

EP_MAX = 1000
EP_LEN = 500

"""Enviroment Setting"""
# GAME = 'Pong_v303'
# env = gym.make(GAME).unwrapped
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.n
"""initialize agents"""
agent = PPONet(A_DIM,S_DIM)
# zen = Yolo()

"""H-parameters"""
all_ep_r = []
MIN_BATCH_SIZE = 64
GAMMA = 0.9

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):
        # env.render()
        """choose bounding box"""
        box_frame = zen.look(s)
        """
        greedy layer:
        n box_frames similar with initial box_frame.
        return: A list like [frame0, frame1, ...]
        """
        frames = zen.greedy_layer(box_frame)
        all_frames = [box_frame] + frames
        new_frames = [np.array([s,f]).transpose((1,2,0)) for f in all_frames]
        """select best one by state value"""
        values = [agent.get_v(f) for f in new_frames]
        best_index = 0
        for i in range(1,len(values)):
            if values[i] > values[best_index]:best_index = i
        best_s = new_frames[best_index]
        """
        Train yolo part:
        loss = Critic(best) - Critic(selecteds)
        """
        loss = values[best_index] - values[0]
        zen.store(s, box_frame, loss)

        """RL's Action"""
        a = agent.choose_action(best_s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append(r)

        s = s_
        ep_r += r

        """
        use trajcectory segment experience update ppo.
        """
        if (t+1) % MIN_BATCH_SIZE == 0 or t == EP_LEN-1 or done:
            v_s_ = agent.get_v(s_)
            if done:v_s_ = 0
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            agent.update(bs, ba, br)

        if done:break

    """
    End of Epoch
    """
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print('Ep: %i' % ep,"|Ep_r: %i" % ep_r)

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()
