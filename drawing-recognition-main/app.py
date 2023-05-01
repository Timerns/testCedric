#Importing modules
import pygame  
import tensorflow as tf  
from PIL import Image 
import numpy as np 
import os  
from tensorflow.python.ops.numpy_ops import np_config  

#Enabling NumPy behavior to prevent TensorFlow from casting NumPy arrays to Tensors
np_config.enable_numpy_behavior()

#Loading the trained CNN model from the saved file
model = tf.keras.models.load_model('final-cnn-digits-model')

#Initializing Pygame and setting the screen dimensions
pygame.init()
screen = pygame.display.set_mode((1000, 500))

#Creating a drawing area on the screen
drawing_area = pygame.Rect(500, 0, 500, 500)

#Drawing a black rectangle on the screen to fill the drawing area
pygame.draw.rect(screen, "black", (500,0,500,500), 0)

#Setting the caption of the screen
pygame.display.set_caption("digits recognition")

#Initializing a variable to keep track of the game state
game_running = True

#Creating a Pygame font object to display text on the screen
my_font = pygame.font.Font(None, 30)

#Creating a list to hold the text surfaces and their corresponding rectangles
text = []

#Creating a text surface and rectangle for the basic text
basic_text_surface = my_font.render('Ai predictions : ', True, "white")
basic_text_rect = basic_text_surface.get_rect(center=(500 / 2, 20))

#Appending the basic text surface and rectangle to the text list
text.append((basic_text_surface,basic_text_rect))

#Creating text surfaces and rectangles for each digit (0-9)
for i in range(10):
    surface = my_font.render(str(i)+" : ", True, "white")
    rect = basic_text_surface.get_rect(center=(100, ((i+1)*30)+10))
    text.append((surface, rect))

#Function to display the text on the screen
def display_text():
    for surface, rect in text:
        screen.blit(surface, rect)

#Function to save the image drawn on the screen
def save_image():
    sub = screen.subsurface(drawing_area)
    pygame.image.save(sub, "image.jpg")

#Function to predict the digit drawn on the screen using the trained CNN model
def predict():
    img = Image.open("image.jpg").convert('L').resize((28, 28))
    img = np.array(img)
    img = img/255
    predictions = model(img[None, :, :]).tolist()
    return predictions

#Function to update the prediction text on the screen
def update_predictions(predictions):
    index = 0
    max_index = np.where(predictions[0] == np.amax(predictions[0]))
    for surface, rect in text[1:]:
        text.remove((surface, rect))
        if index == max_index[0]:
            text_color = "green"
        else:
            text_color = "white"
        surface = my_font.render(str(index) +" : "+ str(round(predictions[0][index] *100, 2))+" %", True, text_color)
        text.append((surface, rect))
        index +=1

#Start a while loop to keep the game running
while game_running:
    #Draw a black rectangle on the left side of the screen
    pygame.draw.rect(screen, "black", (0, 0, 480, 500), 0)
    #Draw a white rectangle on the right side of the screen
    pygame.draw.rect(screen, "white", (480, 0, 20, 500), 0)
    #Call the display_text function to display some text on the screen
    display_text()
    
    #Start a for loop to handle events
    for event in pygame.event.get():
        #If the event is a quit event (clicking the X button on the window)
        if event.type == pygame.QUIT:
            #If the file "image.jpg" exists in the directory
            if os.path.exists("image.jpg"):
                #Remove the file "image.jpg" from the directory
                os.remove("image.jpg")
            #Set the game_running variable to False to exit the while loop
            game_running = False

        #If the event is a mouse button up event
        if event.type == pygame.MOUSEBUTTONUP:
            #Get the position of the mouse
            mouse_position = pygame.mouse.get_pos()
            #If the mouse is within the drawing area
            if drawing_area.collidepoint(mouse_position):
                #Call the save_image function to save the image
                save_image()
                #Call the predict function to get predictions
                predictions = predict()
                #Call the update_predictions function to update the predictions on the screen
                update_predictions(predictions)

    #If the left mouse button is pressed
    if pygame.mouse.get_pressed() == (1, 0, 0):
        # Get the position of the mouse
        mouse_position = pygame.mouse.get_pos()
        # If the mouse is within the drawing area
        if drawing_area.collidepoint(mouse_position):
            # Draw a white circle on the screen at the mouse position
            pygame.draw.circle(surface=screen, center=mouse_position, color="white", radius=12)

    #If the right mouse button is pressed
    if pygame.mouse.get_pressed() == (0, 0, 1):
        #Get the position of the mouse
        mouse_position = pygame.mouse.get_pos()
        #If the mouse is within the drawing area
        if drawing_area.collidepoint(mouse_position):
            #Draw a black circle on the screen at the mouse position
            pygame.draw.circle(surface=screen, center=mouse_position, color="black", radius=12)

    #If the middle mouse button is pressed
    if pygame.mouse.get_pressed() == (0, 1, 0):
        #Get the position of the mouse
        mouse_position = pygame.mouse.get_pos()
        #If the mouse is within the drawing area
        if drawing_area.collidepoint(mouse_position):
            #Draw a black rectangle on the screen to clear the drawing area
            pygame.draw.rect(screen, "black", (500,0,500,500), 0)

    #Update the display to show the changes
    pygame.display.flip()

    
