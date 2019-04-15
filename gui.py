import pygame
import easygui
import fileApi 
import imageProcessing as imgProc

class ocrApp(object):
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('OCR App')

        self.api = fileApi.fileAPI()

        self.layout = 1
        self.WIDTH = 640
        self.HEIGHT = 480
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.ACTIVE_COLOR = pygame.Color('dodgerblue1')
        self.INACTIVE_COLOR = pygame.Color('dodgerblue4')
        self.buttonFont = pygame.font.Font(None, 30)
        self.footerFont = pygame.font.Font(None, 16)
        self.headerFont = pygame.font.Font(None, 50)

    def draw_app_text(self, screen):
        screen.blit(self.footerFont.render('wykonane przez: Kontowicz Piotr, Przybyłowski Paweł, Szkudlarek Damian', True, self.BLACK), (10, self.HEIGHT-20))

    def draw_button(self, button, screen):
        """Draw the button rect and the text surface."""
        pygame.draw.rect(screen, button['color'], button['rect'])
        screen.blit(button['text'], button['text rect'])


    def create_button(self, x, y, w, h, text, callback):
        """A button is a dictionary that contains the relevant data.

        Consists of a rect, text surface and text rect, color and a
        callback function.
        """
        text_surf = self.buttonFont.render(text, True, self.WHITE)
        button_rect = pygame.Rect(x, y, w, h)
        text_rect = text_surf.get_rect(center=button_rect.center)
        button = {
            'rect': button_rect,
            'text': text_surf,
            'text rect': text_rect,
            'color': self.INACTIVE_COLOR,
            'callback': callback,
            }
        return button

    def show_image(self, screen, image, x, y):
        screen.blit(image, (x,y))

    def main(self):
        screen = pygame.display.set_mode((640, 480))
        clock = pygame.time.Clock()
        done = False
        
        number = 0

        def load_files():
            paths = easygui.fileopenbox(title='Choose file', default='*.png', filetypes=['*.png', '*.txt'], multiple=True)
            if paths != None:
                fileNames = self.api.readFiles(paths)
                imgProc.showResizedImage('Result',imgProc.straightenImage(self.api.getImage(fileNames[0])),2)
                #DO POPRAWY WCZYTYWANIE ŚCIEŻKI

        def save_results():
            path = easygui.filesavebox()
            if path != None:
                self.api.saveResults(path)

        def quit_game():  # A callback function for the button.
            nonlocal done
            done = True

        def set_layout_1():
            self.layout = 1

        def set_layout_2():
            self.layout = 2

        buttonLoadFile = self.create_button(100, 100, 180, 60, 'Load file(s)', load_files)
        buttonSave = self.create_button(100, 200, 180, 60, 'Save results', save_results)
        buttonQuit = self.create_button(100, 300, 180, 60, 'Quit', quit_game)
        menuGap = self.create_button(360, 0, 180, 30, None, None)
        menuAppName = self.create_button(540, 0, 100, 30, 'OCR App', None)

        layout1 = self.create_button(0, 0, 180, 30, 'Processing', set_layout_1)
        layout2 = self.create_button(180, 0, 180, 30, 'Verification', set_layout_2)

        buttonLoadImage = self.create_button(100, 100, 180, 60, "Load image file", load_files)
        buttonLoadTextFile = self.create_button(100, 200, 180, 60, "Load text file", load_files)
        buttonCompare = self.create_button(100, 300, 180, 60, "Compare files", None)

        # A list that contains all buttons.
        button_list = [buttonLoadFile, buttonSave, buttonQuit]
        button_list2 = [buttonLoadImage, buttonLoadTextFile, buttonCompare]
        button_menu = [layout1, layout2, menuGap, menuAppName]

        while not done:

            if self.layout == 1:

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
                                if button['rect'].collidepoint(event.pos) and button['callback'] is not None:
                                    button['callback']()
                    elif event.type == pygame.MOUSEMOTION:
                        #sprawdzamy jak poruszlismy myszkiem
                        for button in button_list:
                            #jak wyjechalismy poza obreb przycisku to zmieniamy kolor na nieaktywny i to samo w 2 strone
                            if button['rect'].collidepoint(event.pos):
                                button['color'] = self.ACTIVE_COLOR
                            else:
                                button['color'] = self.INACTIVE_COLOR
                        for button in button_menu:
                            if button['rect'].collidepoint(event.pos) and button['callback'] != None:
                                button['color'] = self.ACTIVE_COLOR
                            else:
                                button['color'] = self.INACTIVE_COLOR
                image = pygame.image.load('data/test.png')

                screen.fill(self.WHITE)
                self.show_image(screen, image, 320, 130)
                self.draw_app_text(screen)
                for button in button_list:
                    self.draw_button(button, screen)
                for button in button_menu:
                    self.draw_button(button, screen)
                pygame.display.update()
                clock.tick(30)
            elif self.layout == 2:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            for button in button_menu:
                                if button['rect'].collidepoint(event.pos) and button['callback'] is not None:
                                    button['callback']()
                    elif event.type == pygame.MOUSEMOTION:
                        for button in button_list2:
                            if button['rect'].collidepoint(event.pos):
                                button['color'] = self.ACTIVE_COLOR
                            else:
                                button['color'] = self.INACTIVE_COLOR
                        for button in button_menu:
                            if button['rect'].collidepoint(event.pos) and button['callback'] != None:
                                button['color'] = self.ACTIVE_COLOR
                            else:
                                button['color'] = self.INACTIVE_COLOR

                screen.fill(self.WHITE)
                self.draw_app_text(screen)
                for button in button_list2:
                    self.draw_button(button, screen)
                for button in button_menu:
                    self.draw_button(button, screen)
                pygame.display.update()
                clock.tick(30)


if __name__ == "__main__":
    app = ocrApp()
    app.main()
    pygame.quit()