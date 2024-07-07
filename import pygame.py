import pygame
import numpy as np
import random

# Inicialización de Pygame
pygame.init()

# Definir constantes del juego
WIDTH, HEIGHT = 640, 480
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_RADIUS = 7

# Crear pantalla del juego
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Ping Pong con Q-learning')

# Definir clases para la pelota y las paletas
class Paddle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

    def move(self, y):
        self.rect.y += y
        if y == -10:
            print("Jugador se mueve arriba")
        elif y == 10:
            print("Jugador se mueve abajo")
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

class Ball:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_RADIUS, BALL_RADIUS)
        self.speed_x = random.choice([-4, 4])
        self.speed_y = random.choice([-4, 4])

    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y = -self.speed_y

    def reset(self):
        self.rect.x = WIDTH // 2
        self.rect.y = HEIGHT // 2
        self.speed_x = random.choice([-4, 4])
        self.speed_y = random.choice([-4, 4])

# Definir la función de Q-learning
class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, ball, paddle):
        return (ball.rect.x, ball.rect.y, ball.speed_x, ball.speed_y, paddle.rect.y)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.actions[np.argmax(self.q_table[state])]

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        action_index = self.actions.index(action)
        predict = self.q_table[state][action_index]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_index] += self.alpha * (target - predict)

# Inicializar objetos del juego y el agente de Q-learning
player_paddle = Paddle(WIDTH - 20)
opponent_paddle = Paddle(10)
ball = Ball()
actions = [-10, 10, 0]
agent = QLearningAgent(actions=actions)

clock = pygame.time.Clock()
running = True
state = agent.get_state(ball, player_paddle)

player_score = 0
opponent_score = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Movimiento del oponente
    if opponent_paddle.rect.centery < ball.rect.y:
        opponent_paddle.move(5)
    elif opponent_paddle.rect.centery > ball.rect.y:
        opponent_paddle.move(-5)

    # Movimiento del jugador
    action = agent.choose_action(state)
    player_paddle.move(action)

    # Mover la pelota
    ball.move()

    # Verificar colisiones
    if ball.rect.colliderect(player_paddle.rect) or ball.rect.colliderect(opponent_paddle.rect):
        ball.speed_x = -ball.speed_x

    # Verificar si la pelota sale de los límites
    if ball.rect.left <= 0:
        ball.reset()
        reward = -1
        player_score += 1
        print("Gano!")
    elif ball.rect.right >= WIDTH:
        ball.reset()
        reward = 1
        opponent_score += 1
        print("Perdio!")
    else:
        reward = 0

    next_state = agent.get_state(ball, player_paddle)
    agent.learn(state, action, reward, next_state)
    state = next_state

    # Dibujar en la pantalla
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, player_paddle.rect)
    pygame.draw.rect(screen, WHITE, opponent_paddle.rect)
    pygame.draw.ellipse(screen, WHITE, ball.rect)
    pygame.draw.aaline(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))

    # Mostrar marcador
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Jugador: {player_score}  Oponente: {opponent_score}', True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()


