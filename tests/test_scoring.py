import math
from flappy_bird_env import FlappyBirdEnv


def test_score_increments_on_pipe_cross():
    env = FlappyBirdEnv(render_mode=False)
    env.reset()

    # Place a single pipe slightly ahead of the bird so we can deterministically
    # step the environment until it passes the bird.
    gap_y = env.WINDOW_HEIGHT // 2 - env.PIPE_GAP // 2
    start_x = env.BIRD_X + env.PIPE_WIDTH + 5
    env.pipes = [{'x': start_x, 'gap_y': gap_y, 'scored': False}]

    initial_score = env.score

    # Compute how many steps are needed for the pipe to move past the bird
    distance_to_move = (start_x + env.PIPE_WIDTH) - env.BIRD_X
    steps_needed = math.ceil(distance_to_move / env.PIPE_VELOCITY) + 1

    # Step the environment keeping the bird safely in the middle (flap occasionally)
    for i in range(steps_needed + 5):
        action = 1 if i % 10 == 0 else 0
        state, reward, done, info = env.step(action)
        if info['score'] > initial_score:
            break

    assert info['score'] == initial_score + 1, f"Expected score {initial_score+1}, got {info['score']}"


if __name__ == '__main__':
    test_score_increments_on_pipe_cross()
    print('Test passed')
