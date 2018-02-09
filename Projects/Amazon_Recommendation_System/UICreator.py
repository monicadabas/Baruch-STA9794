import sys
from PyQt4 import QtGui
import urllib.request
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4 import QtCore
from PIL import Image
import requests
from io import BytesIO

class MyCustomWidget (QtGui.QWidget):
    def __init__ (self, parent = None):
        super(MyCustomWidget, self).__init__(parent)
        self.customQVBoxLayout = QtGui.QVBoxLayout()
        self.textTitle    = QtGui.QLabel()
        self.textRating  = QtGui.QLabel()
        self.textPrice    = QtGui.QLabel()
        self.customQVBoxLayout.addWidget(self.textTitle)
        self.customQVBoxLayout.addWidget(self.textRating)
        self.customQVBoxLayout.addWidget(self.textPrice)        
        self.allQHBoxLayout  = QtGui.QHBoxLayout()
        self.iconQLabel      = QtGui.QLabel()
        self.allQHBoxLayout.addWidget(self.iconQLabel, 0)
        self.allQHBoxLayout.addLayout(self.customQVBoxLayout, 1)
        self.setLayout(self.allQHBoxLayout)


        self.textTitle.setStyleSheet('''
            color: rgb(0, 0, 255);
            font: bold 20px;
        ''')
        self.textRating.setStyleSheet('''
            color: rgb(255, 0, 0);
            font: bold 14px;
        ''')
        self.textPrice.setStyleSheet('''
            color: rgb(255, 0, 0);
            font: bold 14px;
        ''')
        #self.setStyleSheet("margin:5px; border:1px solid rgb(0, 255, 0); ")

    def setTitle (self, text):
        self.textTitle.setText(text)

    def setRating (self, text):
        self.textRating.setText("Rating: " + text)
        
    def setPrice (self, text):
        self.textPrice.setText("Price: " + text + "USD")

    def setIcon (self, imagePath):
        #data = urllib.request.urlopen(imagePath).read()
        #image = QtGui.QImage()
        #image.loadFromData(data)
        response = requests.get(imagePath)
        # QT Creator do not support jpg file format so needed to convert to png
        im = Image.open(BytesIO(response.content))
        im.save('test.png')
        pixmap = QtGui.QPixmap("test.png")
        #print(pixmap)
        self.iconQLabel.setPixmap(pixmap)
        self.iconQLabel.setFixedSize(300,200)
        self.iconQLabel.setScaledContents(True)


class UIQCreator (QtGui.QMainWindow):
    def __init__ (self, item, sim_title):
        super(UIQCreator, self).__init__()
        self.items = QDockWidget("Current Item", self)
        myMyCustomWidget = MyCustomWidget()
        myMyCustomWidget.setTitle(item.title)
        myMyCustomWidget.setRating(str(item.rating))
        myMyCustomWidget.setPrice(str(item.price))
        myMyCustomWidget.setIcon(item.imUrl)
        
        self.items.setWidget(myMyCustomWidget)
        self.items.setFloating(False)
        # Create QListWidget
        self.myQListWidget = QtGui.QListWidget(self)
        """for title, rating, price, icon in [
            ('Title 1', '3.5', '$20',  'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'),
            ('Title 2', '4.4', '$30' , 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'),
            ('Title 3', '2.1', '$60', 'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png')]:"""
        for obj in sim_title:
            title = obj.title
            rating = str(obj.rating)
            price = str(obj.price)
            icon = obj.imUrl
            # Create MyCustomWidget
            myMyCustomWidget = MyCustomWidget()
            
            #myMyCustomWidget.setStyleSheet("border:3px solid rgb(0, 0, 0); ")
            myMyCustomWidget.setTitle(title)
            myMyCustomWidget.setRating(rating)
            myMyCustomWidget.setPrice(price)
            myMyCustomWidget.setIcon(icon)
            # Create QListWidgetItem
            myQListWidgetItem = QtGui.QListWidgetItem(self.myQListWidget)
            # Set size hint
            myQListWidgetItem.setSizeHint(myMyCustomWidget.sizeHint())
            # Add QListWidgetItem into QListWidget
            self.myQListWidget.addItem(myQListWidgetItem)
            self.myQListWidget.setItemWidget(myQListWidgetItem, myMyCustomWidget)
            self.addDockWidget(Qt.TopDockWidgetArea, self.items)
            self.setCentralWidget(self.myQListWidget)
            
def initializeUI (item, sim_title):
    #print("Item is:"+str(item))
    #print("sim_title is:"+str(sim_title))  
    #app = QtGui.QApplication(sys.argv)
    app = QtGui.QApplication(sys.argv)
    QtCore.QCoreApplication.addLibraryPath("/Users/monicadabas/anaconda3/pkgs/pyqt-4.11.4-py35_3")

    print(QImageReader.supportedImageFormats())
    #app = QtGui.QApplication([])
    window = UIQCreator(item, sim_title)
    window.show()
    sys.exit(app.exec_())             
