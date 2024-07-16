import cv2
import numpy as np
import ipywidgets.widgets as widgets
from IPython.display import display, clear_output
from jetbot import Robot, Camera
import time
from flask import Flask, jsonify, request
import torch
import torch.nn as nn

app = Flask(__name__)

robot = Robot()
try:
    camera = Camera.instance(width=224, height=224)
    if(camera):
        print("Camera is initialized")
finally:
    pass

park_number = 6
is_line_ok = False
line_error_count = 0
is_parked =False
linear_speed = 0.11


jetbot_status = {
    'jetbot_status': 'waiting',
    'battery_level': 80,
    'connection_status': 'not connected'
}

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64*30*30, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64*30*30)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Modeli Yükleme
model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn.pth'))
model.eval()

def detect_plate():
    total_predictions = []
    try:
        global park_number
        # 30 elemanlı total predictions listesi
        print("deneme")
        total_predictions.append(detector("left", 95))    
        total_predictions.append(detector("left", 3))
        total_predictions.append(detector("left", 3))
        total_predictions.append(detector("left", 3))
        total_predictions.append(detector("left", 3))
        
        total_predictions.append(detector("right", 3))
        total_predictions.append(detector("right", 3))
        total_predictions.append(detector("right", 3))
        total_predictions.append(detector("right", 3))
        #turn_robot_left(12.0)

        final_prediction = is_more_than_40(total_predictions, park_number)
        print(final_prediction)
        if final_prediction == True:
            # Park yeri bulunduysa yapılacaklar.
            #park_robot()
            return {'status': 'parked'}

        else:
            total_predictions.clear()
            # Park yeri bulanamadıysa yapılacaklar
            print("deneme ")
            turn_robot_right(90)
            total_predictions.append(detector("right", 95))
            total_predictions.append(detector("right", 3))
            total_predictions.append(detector("right", 3))
            total_predictions.append(detector("right", 3))
            total_predictions.append(detector("right", 3))
            
            total_predictions.append(detector("left", 3))
            total_predictions.append(detector("left", 3))
            total_predictions.append(detector("left", 3))
            total_predictions.append(detector("left", 3))
            #turn_robot_right(12.0)

            final_prediction = is_more_than_40(total_predictions, park_number)

            if final_prediction == True:
                # Park yeri bulunduysa yapılacaklar
                # park_robot()
                return {'status': 'parked'}
            else:
                # Park yeri bulunmadıysa yapılacaklar
                turn_robot_left(90)
                return {'status': total_predictions}

    except Exception as e:
        return {'status': 'error'}

    finally:
        stop_camera()
        
def detector(turn_rotation, turn_angle):
    global camera
    if turn_rotation == "right":
        turn_robot_right(turn_angle)
    elif turn_rotation == "left":
        turn_robot_left(turn_angle)
    
    predictions = []
    try:
        camera = start_camera()
        count = 0
        while count < 10:
            image = camera.value
            prediction, _ = preprocess_and_predict_image(image, model)
            predictions.append(prediction)
            print("prediction: ", str(prediction))
            count += 1
    finally:
        stop_camera()
    return predictions

def start_camera():
    global camera
    if not camera:
        camera = Camera.instance(width=224, height=224)
    else:
        camera.stop()
        camera = Camera.instance(width=224, height=224)
    return camera

def stop_camera():
    global camera
    if camera:
        camera.stop()

# Zoom fonksiyonu
def zoom_center(image, zoom_factor=1):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    radius_x, radius_y = width // (2 * zoom_factor), height // (2 * zoom_factor)
    
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y
    
    cropped = image[min_y:max_y, min_x:max_x]
    return cv2.resize(cropped, (width, height))

# Görüntüyü işleme ve tahmin yapma fonksiyonu
def preprocess_and_predict_image(image, model):
    # Görüntüyü ortasından zoom yap
    zoomed_image = zoom_center(image, zoom_factor=1)
    # Görüntüyü gri tonlamalıya çevir
    gray_image = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2GRAY)
    # Binary threshold uygulama
    _, binary_image = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
    # Görüntüyü yeniden boyutlandır
    resized_image = cv2.resize(binary_image, (128, 128))
    # Görüntüyü Tensora çevir
    tensor_image = torch.tensor(resized_image).unsqueeze(0).unsqueeze(0).float() / 255.0
    
    # Model ile tahmin yap
    outputs = model(tensor_image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item(), binary_image

# Fonksiyon: Görüntüyü işle ve siyah çizgiyi tespit et
def detect_black_line(image):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü genişlik ve yükseklikten kırp
    height, width = gray.shape
    crop_img = gray[int(height * 0.4):height, int(width * 0.35):int(width * 0.65)]  # Alt %60 ve yanlardan %30
    
    # Siyah çizgiyi maskelemek için bir eşik değeri kullan
    _, binary = cv2.threshold(crop_img, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Görüntünün merkezindeki siyah çizginin konumunu bul
    height, width = binary.shape
    crop_img = binary  # Alt %60 ve yanlardan %30'u alındı
    
    # Konturların tespiti
    contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturu seç
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Orijinal genişlik boyutuna geri dönmek için cx'i genişliğin %30'u kadar kaydır
            cx += int(width * 0.35)
            return cx, cy, crop_img  # İşlenmiş alt %60'lık ve yanlardan %30'luk kısmı geri döndür
    return None, None, crop_img

# Fonksiyon: Çizgiyi takip et ve görüntüyü göster
def update_image(change):
    global is_line_ok
    global linear_speed
    global donus_count
    image = change['new']
    cx, cy, binary = detect_black_line(image)
    
    if cx is not None:
        is_line_ok = True
        width = image.shape[1]
        
        # Hata hesapla (çizgi merkezden ne kadar uzak?)
        error = 57 - cx
        
        print("error : ",str(error))
        # Hata oranına göre robotu kontrol et
        Kp = 0.0011  # Hata katsayısı, dönüş hızını ayarlamak için kullanılacak  
        angular_speed = Kp * error  # Hata ile orantılı dönüş hızı

        # Dönüş hızlarını sınırlamak için bir maksimum dönüş hızı belirleyelim
        max_angular_speed = 0.2  # Dönüş hızını azaltarak daha hassas kontrol sağlandı
        angular_speed = max(-max_angular_speed, min(max_angular_speed, angular_speed))

        # Hatanın büyüklüğüne göre dönüş hızını kademeli olarak ayarla
        if abs(error) < 10:
            # Hata çok küçükse, düz ilerle
            robot.set_motors(linear_speed - angular_speed * 0.2, linear_speed + angular_speed * 0.2)
        elif abs(error) < 20:
            # Hata küçükse, hafif düzeltme yap
            robot.set_motors(linear_speed - angular_speed * 0.5, linear_speed + angular_speed * 0.5)
        else:
            # Hata büyükse, belirgin dönüş yap
            robot.set_motors(linear_speed - angular_speed, linear_speed + angular_speed)
    else:
        is_line_ok = False
        linear_speed = 0.09
        donus_count = 0 
        print("Line not detected")


def detect_park_number_for_park(image):
    # Görüntüyü gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Görüntüyü genişlik ve yükseklikten kırp
    height, width = gray.shape
    crop_img = gray[:int(height * 0.6), int(width * 0.35):int(width * 0.65)] #üst yüzde 60'ı al^yanlardan yüzde 35 kes

    
    # Siyah çizgiyi maskelemek için bir eşik değeri kullan
    _, binary = cv2.threshold(crop_img, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Görüntünün merkezindeki siyah çizginin konumunu bul
    height, width = binary.shape
    crop_img = binary  # Alt %60 ve yanlardan %30'u alındı
    
    # Konturların tespiti
    contours, _ = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # En büyük konturu seç
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Orijinal genişlik boyutuna geri dönmek için cx'i genişliğin %30'u kadar kaydır
            cx += int(width * 0.35)
            return cx, cy, crop_img  # İşlenmiş alt %60'lık ve yanlardan %30'luk kısmı geri döndür
    return None, None, crop_img


def update_image_for_park(change):
    global is_line_ok
    global linear_speed
    global donus_count
    
    image = change['new']
    cx, cy, binary = detect_park_number_for_park(image)
    linear_speed = 0.085
    
    if cx is not None:
        is_line_ok = True
        width = image.shape[1]
        print("error cx park :",str(cx))
        # Hata hesapla (çizgi merkezden ne kadar uzak?)
        error = 57 - cx
        
        # Hata oranına göre robotu kontrol et
        Kp = 0.0011  # Hata katsayısı, dönüş hızını ayarlamak için kullanılacak  
        angular_speed = Kp * error  # Hata ile orantılı dönüş hızı

        # Dönüş hızlarını sınırlamak için bir maksimum dönüş hızı belirleyelim
        max_angular_speed = 0.2  # Dönüş hızını azaltarak daha hassas kontrol sağlandı
        angular_speed = max(-max_angular_speed, min(max_angular_speed, angular_speed))

        # Hatanın büyüklüğüne göre dönüş hızını kademeli olarak ayarla
        if abs(error) < 10:
            # Hata çok küçükse, düz ilerle
            robot.set_motors(linear_speed - angular_speed * 0.2, linear_speed + angular_speed * 0.2)
        elif abs(error) < 20:
            # Hata küçükse, hafif düzeltme yap
            robot.set_motors(linear_speed - angular_speed * 0.5, linear_speed + angular_speed * 0.5)
        else:
            # Hata büyükse, belirgin dönüş yap
            robot.set_motors(linear_speed - angular_speed, linear_speed + angular_speed)
    else:
        is_line_ok = False
        linear_speed = 0.09
        donus_count = 0 
        print("Line not detected")

# Fonksiyon: Robotu sağa döndür
def turn_robot_right(degrees):
    global robot
    turn_time = degrees / 360.0  # 360 derece için 1 saniye varsayımı
    robot.right(speed=0.15)  # Robotu sağa döndürmek için sol motoru çalıştırıyoruz
    time.sleep(turn_time * 2.15)
    robot.stop()
    
# Fonksiyon: Robotu sola döndür
def turn_robot_left(degrees):
    global robot
    turn_time = degrees / 360.0  # 360 derece için 1 saniye varsayımı
    robot.left(speed=0.15)  # Robotu sağa döndürmek için sol motoru çalıştırıyoruz
    time.sleep(turn_time * 2.15)
    robot.stop()

# Fonksiyon: Robotu park et
def park_robot():
    global robot
    robot.forward(speed=0.3)
    time.sleep(0.9)
    robot.stop()

def robot_forward_for_park():
    global robot
    robot.forward(speed =0.3)
    time.sleep(0.6)
    robot.stop()
    print("park icin ileri durdu")

# Fonksiyon: İki boyutlu listeyi kontrol et
def is_more_than_40(iki_boyutlu_liste, sayi):
    # İki boyutlu listeyi tek boyutlu bir listeye dönüştür
    tek_boyutlu_liste = [eleman for alt_liste in iki_boyutlu_liste for eleman in alt_liste]
    
    # Listenin toplam eleman sayısını bul
    toplam_eleman_sayisi = len(tek_boyutlu_liste)
    
    # Verilen sayının frekansını say
    sayi_frekansi = tek_boyutlu_liste.count(sayi)
    
    # Yüzde 10'tan fazla olup olmadığını kontrol et
    if sayi_frekansi > (0.1 * toplam_eleman_sayisi):
        return True
    else:
        return False

@app.route('/get_jetbot_status', methods=['GET'])
def get_jetbot_status():
    global jetbot_status
    jetbot_status["connection_status"] = "connected"
    return jsonify(jetbot_status)

@app.route('/park', methods=['POST'])
def detect():
    if jetbot_status["jetbot_status"] in ["waiting", "parked"]:
        global donus_count
        donus_count = 0
        jetbot_status["jetbot_status"] = "parking"
        global camera, robot, is_line_ok  # Bu satırı ekleyin
        global park_number
        global first_follow
        global linear_speed
        
        linear_speed = 0.11
        data = request.get_json()
        park_number = data.get('park_number')
        
        print("park number: ", str(park_number))
        first_follow = True
        # Kameradan alınan görüntü değiştiğinde update_image fonksiyonunu çağır
        start_camera()
        camera.observe(update_image, names='value')
        
        # Uygulama çalışırken diğer kodları çalıştırmak için bir iş parçacığı kullan
        try:
            start_park_time = time.time()
            while (time.time() - start_park_time) < 180:
                print("is line ok: ", str(is_line_ok))
                print("linear speed : ",str(linear_speed))
                time.sleep(0.05)
                if first_follow == True:
                    # 2 saniyelik kesiciye girme
                    if is_line_ok != True:
                        # Yol yoksa dönsün
                        turn_robot_right(185)
                        time.sleep(1)
                        first_follow = False

                else:
                    # 2 saniyede bir observe etmeyi ve kamerayı stopla 90 derece dön kameradan oku eşleme yap 
                    if is_line_ok:
                        if(donus_count == 0):
                            linear_speed = 0.09
                            time.sleep(0.2)
                        else:
                            linear_speed = 0.1
                            time.sleep(0.2)
                        start_time = time.time()
                        while (time.time() - start_time) < 0.85:
                            time.sleep(0.1)
                        camera.unobserve(update_image, names='value')
                        donus_count = donus_count + 1
                        
                        stop_camera()
                        robot.stop()
                        sonuc = detect_plate()
                        if sonuc["status"] == "parked":
                            # park etme işlemleri yapılacak ileri biraz gidecek ve sonrasında 1 saniyeliğine camera observe(update_image,name='value') yapılacak linear_speed 0.09 olarak yapılacak
                            
                            print("observe moduna girildi")
                            robot_forward_for_park()
                            start_park_robot_to_number_time = time.time()
                            time.sleep(1)
                            start_camera()
                            linear_speed = 0.09
                            camera.observe(update_image,names='value')
                            time.sleep(1.7)
                            #camera.unobserve(update_image_for_park,names='value')
                            
                            #camera.unobserve(update_image_for_park,names='value')
                            #camera.unobserve(update_image,names='value')
                            stop_camera()
                            robot.stop()
                            # Park etme işlemi başarılı.
                            jetbot_status["jetbot_status"] = "parked"
                            return 'park edildi.'
                        
                        else:
                            print(sonuc)
                            start_camera()

                        camera.observe(update_image, names='value')
                        time.sleep(0.5)

                    else:
                        turn_robot_right(165)
                        time.sleep(1)
                        first_follow = False

            jetbot_status["jetbot_status"] = "park not found"
            first_follow = True
            linear_speed =0.11
            return
            

        except KeyboardInterrupt:
            robot.stop()
        finally:
            if(jetbot_status["jetbot_status"] == "parked"):
                camera.unobserve(update_image, names='value')
            else:
                camera.unobserve(update_image, names='value')
            stop_camera()
            robot.stop()
    

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=4646)
    except KeyboardInterrupt:
        robot.stop()
    finally:
        camera.unobserve(update_image, names='value')
        stop_camera()
        robot.stop()
