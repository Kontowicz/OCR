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
    #screen.blit(headerFont.render('OCR APP', True, BLACK), (WIDTH/2-80, 20))
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

def show_image(screen, image, x, y):
    screen.blit(image, (x,y))

layout = 1
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

    def set_layout_1():
        global layout
        layout = 1

    def set_layout_2():
        global layout
        layout = 2

    buttonLoadFile = create_button(100, 100, 180, 60, 'Load file(s)', load_files)
    buttonSave = create_button(100, 200, 180, 60, 'Save results', save_results)
    buttonQuit = create_button(100, 300, 180, 60, 'Quit', quit_game)
    menuGap = create_button(360, 0, 180, 30, None, None)
    menuAppName = create_button(540, 0, 100, 30, 'OCR App', None)

    layout1 = create_button(0, 0, 180, 30, 'Processing', set_layout_1)
    layout2 = create_button(180, 0, 180, 30, 'Verification', set_layout_2)

    buttonLoadImage = create_button(100, 100, 180, 60, "Load image file", None)
    buttonLoadTextFile = create_button(100, 200, 180, 60, "Load text file", None)
    buttonCompare = create_button(100, 300, 180, 60, "Compare files", None)
    # A list that contains all buttons.
    button_list = [buttonLoadFile, buttonSave, buttonQuit]
    button_list2 = [buttonLoadImage, buttonLoadTextFile, buttonCompare]
    button_menu = [layout1, layout2, menuGap, menuAppName]

    while not done:

        while layout == 1 and not done:

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
                        for button in button_menu:
                            if button['rect'].collidepoint(event.pos):
                                button['callback']()
                elif event.type == pygame.MOUSEMOTION:
                    #sprawdzamy jak poruszlismy myszkiem
                    for button in button_list:
                        #jak wyjechalismy poza obreb przycisku to zmieniamy kolor na nieaktywny i to samo w 2 strone
                        if button['rect'].collidepoint(event.pos):
                            button['color'] = ACTIVE_COLOR
                        else:
                            button['color'] = INACTIVE_COLOR
                    for button in button_menu:
                        if button['rect'].collidepoint(event.pos) and button['callback'] != None:
                            button['color'] = ACTIVE_COLOR
                        else:
                            button['color'] = INACTIVE_COLOR
            image = pygame.image.load('data/random.png')

            screen.fill(WHITE)
            show_image(screen, image, 320, 130)
            draw_app_text(screen)
            for button in button_list:
                draw_button(button, screen)
            for button in button_menu:
                draw_button(button, screen)
            pygame.display.update()
            clock.tick(30)

        while layout == 2 and not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for button in button_menu:
                            if button['rect'].collidepoint(event.pos):
                                button['callback']()
                elif event.type == pygame.MOUSEMOTION:
                    for button in button_list2:
                        if button['rect'].collidepoint(event.pos):
                            button['color'] = ACTIVE_COLOR
                        else:
                            button['color'] = INACTIVE_COLOR
                    for button in button_menu:
                        if button['rect'].collidepoint(event.pos) and button['callback'] != None:
                            button['color'] = ACTIVE_COLOR
                        else:
                            button['color'] = INACTIVE_COLOR

            screen.fill(WHITE)
            draw_app_text(screen)
            for button in button_list2:
                draw_button(button, screen)
            for button in button_menu:
                draw_button(button, screen)
            pygame.display.update()
            clock.tick(30)

main()
pygame.quit()