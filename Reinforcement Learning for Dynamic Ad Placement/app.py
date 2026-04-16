import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

st.title("Reinforcement Learning for Ad Placement")

times = ["morning", "afternoon", "evening"]
devices = ["mobile", "desktop"]
interests = ["sports", "tech", "fashion"]

actions = ["sports_ad", "tech_ad", "fashion_ad"]

if "Q" not in st.session_state:
    Q = {}
    for t in times:
        for d in devices:
            for i in interests:
                state = (t, d, i)
                Q[state] = {a: 0 for a in actions}
    st.session_state.Q = Q

if "epsilon" not in st.session_state:
    st.session_state.epsilon = 0.5

alpha = 0.1


def get_reward(state, action):
    _, _, interest = state
    if action == interest + "_ad":
        return np.random.choice([0, 1], p=[0.4, 0.6])
    else:
        return np.random.choice([0, 1], p=[0.6, 0.4])


def choose_action(state):
    if random.uniform(0, 1) < st.session_state.epsilon:
        return random.choice(actions)
    else:
        return max(st.session_state.Q[state], key=st.session_state.Q[state].get)


episodes = st.slider("Episodes", 100, 10000, 3000)

if st.button("Train Agent"):
    rewards = []

    for ep in range(episodes):
        state = (random.choice(times), random.choice(devices), random.choice(interests))
        action = choose_action(state)
        reward = get_reward(state, action)

        st.session_state.Q[state][action] = st.session_state.Q[state][
            action
        ] + alpha * (reward - st.session_state.Q[state][action])

        rewards.append(reward)

        st.session_state.epsilon = max(0.01, st.session_state.epsilon * 0.995)

    st.success("Training completed!")

    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Reward Over Time")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

st.subheader("Test Best Ad Strategy")

time_input = st.selectbox("Time", times)
device_input = st.selectbox("Device", devices)
interest_input = st.selectbox("Interest", interests)

state = (time_input, device_input, interest_input)

if st.button("Get Best Ad"):
    best_ad = max(st.session_state.Q[state], key=st.session_state.Q[state].get)
    st.write(f"Best Ad for {state}: {best_ad}")
