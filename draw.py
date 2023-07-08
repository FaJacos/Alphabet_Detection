import pygame
import cv2 as cv
import ctypes
import main

ctypes.windll.shcore.SetProcessDpiAwareness(True)

pygame.init()
pygame.font.init()

run = True
fps = 3000
timer = pygame.time.Clock()
WIDTH, HEIGHT = 700, 800
screen = pygame.display.set_mode([WIDTH, HEIGHT])
font = pygame.font.Font(None,30)

paint_size = 15

def draw_menu():
    pygame.draw.rect(screen, 'gray',[0,0, WIDTH, 100])
    
    clear_canvas = pygame.draw.rect(screen, 'white', [10,10,140,80])
    pygame.draw.rect(screen, 'black', [10,10,140,80], 3)
    screen.blit(font.render("Clear", True, 'black'), (55, 40))
    
    guess_box = pygame.draw.rect(screen, 'white', [160,10,240,80])
    pygame.draw.rect(screen, 'black', [160,10,240,80],3)
    
    correct = pygame.draw.rect(screen,'green', [410,10,90,80])
    pygame.draw.rect(screen, 'black', [410,10,90,80],3)
    screen.blit(font.render("Correct", True, 'black'), (420, 40))
    
    wrong = pygame.draw.rect(screen,'red', [510,10,90,80])
    pygame.draw.rect(screen, 'black', [510,10,90,80],3)
    screen.blit(font.render("Wrong", True, 'black'), (523, 40))
    
    emoji = pygame.draw.rect(screen,'white',[610,10,80,80])
    pygame.draw.rect(screen, 'black', [610,10,80,80],3)
    
    pygame.draw.rect(screen, 'black' ,[0,0,WIDTH,HEIGHT],3)
    pygame.draw.line(screen, 'black', (0,100), (WIDTH, 100), 3)
    
    button_list = [clear_canvas, guess_box, correct, wrong, emoji]
    
    return button_list

canvas_size = [700,700]
canvas = pygame.Surface(canvas_size)
canvas.fill((255,255,255))
    
while run:
    screen.fill('white')
    
    button_list = draw_menu()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            for button in range(len(button_list)):
                if button_list[button].collidepoint(event.pos):
                    if button == 0:
                        canvas.fill('white')
    
    x, y = screen.get_size()
    screen.blit(canvas,[0,100,WIDTH,HEIGHT])
    
    mx, my = pygame.mouse.get_pos()
    left_click = pygame.mouse.get_pressed()[0]

    if left_click and my > 100:
        pygame.draw.circle(canvas, 'black', [mx, my-100], paint_size)
    
    pygame.image.save(screen.subsurface([0,100,700,700]),'drawnImage.PNG')
    print(main.eval())
    
    pygame.display.flip()
    timer.tick(fps)

pygame.quit()