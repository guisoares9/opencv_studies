import math
import pygame
import pickle
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from multiprocessing import Process, Queue

from pygame.locals import *
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# IMPORT OBJECT LOADER
# módulo que carrega arquivos .obj em OpenGL no python
from objloader import *


from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()

def cvt_coord_cv2gl(opencv_pixel, p_angle, z, image_size):
    """
    Converte as coordenadas de um ponto no OpenCV para um ponto em pygame OpenGL
    opencv_pixel((x, y))  -> coordenadas do ponto na imagem do openCV
    p_angle(float) -> ângulo de perspectiva da janela do OpenGl em graus
    z(float)  -> distância z EM RELAÇÃO À CÃMERA do ponto final
    image_size(x, y)-> tamanho da image do OpenCV em pixels
    retorno (x, y) coordenadas do ponto no sistema openGL
    """
    #converte vamores de image w e h para o sistema de coordenadas do pygameOpenGL
    #   OPENCV                 OPENGL_pygame
    # __________              ___________
    #| o------> |            |    ^      |
    #| |        |            |    |      |
    #| |        |            |    o--->  |
    #| \/       |            |           |
    #|__________|            |___________|
    pygame_x = -opencv_pixel[1] + image_size[1]/2
    pygame_y =  opencv_pixel[0] - image_size[0]/2
    k= 2 * z * math.tan(math.radians(p_angle/2)) / image_size[1]
    return pygame_y*k, pygame_x*k


def get_screen_dimensions(perspective_angle, distance_to_camera):
    half_height= 2*(distance_to_camera)*math.tan(math.radians(perspective_angle/2))/2
    half_width=  half_height*4/3
    return half_width, half_height


def draw_background(width, height, z_axis):
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0); glVertex3f(-width, -height,  z_axis)
    glTexCoord2f(1.0, 0.0); glVertex3f( width, -height,  z_axis)
    glTexCoord2f(1.0, 1.0); glVertex3f( width,  height,  z_axis)
    glTexCoord2f(0.0, 1.0); glVertex3f(-width,  height,  z_axis)
    glEnd()


def init(perspective_angle):
    pygame.init()
    pygame.display.set_mode((640*2, 480*2), DOUBLEBUF | OPENGL)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glShadeModel(GL_SMOOTH)
    glLoadIdentity()
    gluPerspective(perspective_angle, 640/480, 0.1, 100.0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTranslatef(0, 0, -15)

def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)


def main():
    angle = 45.00

    # Inicialização da janela do pygame com suporte à OPENGL
    init(angle)

    # Carrega os modelos dos objetos
    obj_dog = OBJ('brain-simple-mesh1.obj', swapyz=True)

    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Carrega as matrizes de calibração da câmera que estão salvas em um arquivo
    mtx =   np.array([[725.7657025001567, 0.0, 302.64698191622074],
        [0.0, 733.3294790963405, 277.5564048217287],
        [0.0, 0.0, 1.0] 
        ])

    dist =  np.array([[0.35136313880105813, -3.2814702001594096, -0.002180299088962941, 0.00383256440875228, 12.160646308561365 ]])

    # corners points of the printed ARuco marker
    # O tamanho aproximado do aruco é de 8.68 cm, e foi escolhido o centro do aruco como origem do sistema de coordenadas
    obj_points = np.array([[-4.34, -4.34, -4.34],
                           [ 4.34, -4.34, -4.34],
                           [ 4.34,  4.34, -4.34],
                           [-4.34,  4.34, -4.34]], dtype= np.float32)

    # Captura um dicionário de arucos com seus IDs
    dict_aruco = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))
    
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()            

    # Loop principal
    while True:
        #time.sleep(1)
        # Limpa todo o conteudo da tela
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # Captura uma imagem da camera
        frame_got, frame = cap.read()

        if frame_got is False:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Converte a imagem em uma textura do OpenGL
        gl_image = Image.fromarray(frame)
        glGenTextures(1)
        glTexImage2D(GL_TEXTURE_2D, 0, 3,
                     gl_image.size[0], gl_image.size[1],
                     0, GL_RGBA, GL_UNSIGNED_BYTE,
                     gl_image.tobytes("raw", "BGRX", 0, -1))

        # Desenha a textura na janela para que o usuario veja
        glEnable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)
        w, h = get_screen_dimensions(angle, 20)
        draw_background(w, h, -5)
        glDisable(GL_TEXTURE_2D)

        glEnable(GL_LIGHTING)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(0,0,0,0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (GLfloat*4)(0,0,0,0))
        glEnable(GL_LIGHT0)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # Feed frame to image queue.
        img_queue.put(frame.copy())        

        # Get face from box queue.
        facebox = box_queue.get()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Procura por arucos na imagem e retornas os Ids e os cantos dos arucos encontrados
        corners, ids, _ = cv2.aruco.detectMarkers(frame_gray, dict_aruco, parameters=parameters)
        
        # caso algum aruco seja encontrado
        if corners != []:
            for corner, ID in zip(corners, ids):
                corners = np.array(corner).astype(np.float32)
                corners = corners.reshape((1, -1, 2))

                # encontra os vetores de rotação e translação que relaciona o sistema de coordenadas da imagem OpenCV
                # e o sistema de coordenadas com o centro no aruco
                ret, vec_rotation, vec_translation = cv2.solvePnP(obj_points, corners, mtx, dist)
                origin = tuple(np.mean(corners[0], axis=0).ravel())
                print(vec_translation)

                # Converte os dados do sistema de coordenadas OpenCV para OpenGL
                x, y = cvt_coord_cv2gl((origin[0],origin[1]),angle, 5, (640,480))

                # Faz o tamanho do objeto desenhado ser proporcial à distância do objeto à câmera,
                # isto é, ajusta a escala do objeto.
                s = 15/np.round(vec_translation[2], decimals=2)

                # Desenha algo em cima do aruco dependendo do ID do aruco encontrado
                glPushMatrix()
                print(ID)
                if ID == 0:
                    # Desenha o objeto cachorro
                    glTranslatef(x, y, 10)
                    vec_rotation = np.round(vec_rotation, decimals=2)
                    r = list(vec_rotation.reshape(1, 3))
                    glRotate(np.rad2deg(np.linalg.norm(r)), r[0][0], -r[0][1], -r[0][2])
                    glScale(s, s, s)
                    glCallList(obj_dog.gl_list)

                glPopMatrix()
            
        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks([face_img])
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            mark_detector.draw_marks(
                 frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            #mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose_r, pose_t = pose_estimator.solve_pose_by_68_points(marks)
                        
            #Stabilize the pose. KALMAN FILTER
            steady_pose = []
            pose_np = np.array([[pose_r],[pose_t]]).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            pose_r = steady_pose[0,:]
            pose_t = steady_pose[1,:]

            point = np.float32([0, -55, -75])
            ctred_face_pts,_ = cv2.projectPoints(point, pose_r, pose_t, mtx, dist)
            origin = np.array([ctred_face_pts[0][0][0], ctred_face_pts[0][0][1]])
            
            # Converte os dados do sistema de coordenadas OpenCV para OpenGL
            x1, y1 = cvt_coord_cv2gl((origin[0],origin[1]),angle, 5, (640,480))

            # Faz o tamanho do objeto desenhado ser proporcial à distância do objeto à câmera,
            # isto é, ajusta a escala do objeto.
            s = 160/np.round(pose_t[2], decimals=2)
            
            glPushMatrix()

            # Desenha o objeto cachorro
            glTranslatef(x1, y1, 10)
            pose_r = np.round(pose_r, decimals=2)
            r1 = list(pose_r.reshape(1, 3))

            #r1[0][0]+=-0*3.14
            #r1[0][1]+=0/2*3.14
            #r1[0][2]+=0*3.14
            glRotate(np.rad2deg(np.linalg.norm(r1)), r1[0][0], -r1[0][1], -r1[0][2])
            glRotate(-90,1,0,0)
            glRotate(-90,0,0,1)
            
            glScale(s, s, s)
            glCallList(obj_dog.gl_list)

            glPopMatrix()

        # Atualiza a tela do pygame para que mostre tudo o que foi processado
        # Senão chamar essa função, não é mostrado nada
        pygame.display.flip()

        # Condição de saída do loop: apertar o [X] da janela
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


if __name__ == '__main__':
    main()
    exit(0)

