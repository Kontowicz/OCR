import pygame
import easygui
import fileApi 

global api
api = fileApi.fileAPI()

pygame.init()

pygame.display.set_caption('OCR App')

WIDTH = 640
HEIGHT = 480

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ACTIVE_COLOR = pygame.Color('dodgerblue1')
INACTIVE_COLOR = pygame.Color('dodgerblue4')
buttonFont = pygame.font.Font(None, 30)
footerFont = pygame.font.Font(None, 16)
headerFont = pygame.font.Font(None, 50)


def draw_app_text(screen):
    screen.blit(headerFont.render('OCR APP', True, BLACK), (WIDTH/2-80, 20))
    screen.blit(footerFont.render('wykonane przez: Kontowicz Piotr, Przybyłowski Paweł, Szkudlarek Damian', True, BLACK), (10, HEIGHT-20))

def draw_button(button, screen):
    """Draw the button rect and the text surface."""
    pygame.draw.rect(screen, button['color'], button['rect'])
    screen.blit(button['text'], button['text rect'])


def create_button(x, y, w, h, text, callback):
    """A button is a dictionary that contains the relevant data.

    Consists of a rect, text surface and text rect, color and a
    callback function.
    """
    text_surf = buttonFont.render(text, True, WHITE)
    button_rect = pygame.Rect(x, y, w, h)
    text_rect = text_surf.get_rect(center=button_rect.center)
    button = {
        'rect': button_rect,
        'text': text_surf,
        'text rect': text_rect,
        'color': INACTIVE_COLOR,
        'callback': callback,
        }
    return button


def main():
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    done = False
    
    number = 0

    def load_files():
        paths = easygui.fileopenbox(title='Chose file', default='*.txt', filetypes=['*.png', '*.txt'], multiple=True)
        api.readFiles(paths)

    def save_results(): 
        path = easygui.filesavebox()   
        api.saveResults(path)


    def quit_game():  # A callback function for the button.
        nonlocal done
        done = True

    buttonLoadFile = create_button(100, 100, 180, 60, 'Load file(s)', load_files)
    buttonSave = create_button(100, 200, 180, 60, 'Save results', save_results)
    buttonQuit = create_button(100, 300, 180, 60, 'Quit', quit_game)
    # A list that contains all buttons.
    button_list = [buttonLoadFile, buttonSave, buttonQuit]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    #sprawdzamy czy lewy klik
                    for button in button_list:
                        #jesli ta to ktory button
                        if button['rect'].collidepoint(event.pos):
                            #a tutaj callback funkcja batona
                            button['callback']()
            elif event.type == pygame.MOUSEMOTION:
                #sprawdzamy jak poruszlismy myszkiem
                for button in button_list:
                    #jak wyjechalismy poza obreb przycisku to zmieniamy kolor na nieaktywny i to samo w 2 strone
                    if button['rect'].collidepoint(event.pos):
                        button['color'] = ACTIVE_COLOR
                    else:
                        button['color'] = INACTIVE_COLOR

        screen.fill(WHITE)
        draw_app_text(screen)
        for button in button_list:
            draw_button(button, screen)
        pygame.display.update()
        clock.tick(30)


main()
pygame.quit()