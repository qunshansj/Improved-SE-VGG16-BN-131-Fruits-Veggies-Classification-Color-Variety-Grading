python


class FruitDetector:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.json_path = './class_indices.json'
        self.weights_path = "./vgg16Net.pth"
        self.class_indict = None
        self.model = None

    def load_class_indict(self):
        assert os.path.exists(self.json_path), "file: '{}' dose not exist.".format(self.json_path)
        with open(self.json_path, "r") as f:
            self.class_indict = json.load(f)

    def load_model(self):
        assert os.path.exists(self.weights_path), "file: '{}' dose not exist.".format(self.weights_path)
        self.model = vgg(model_name="vgg16", num_classes=131).to(self.device)
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        self.model.eval()

    def detect_image(self, img_path):
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        img = self.data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        plt.title(print_res)
        plt.savefig('./save.png', format='png')
        show = cv2.imread('./save.png')
        return print_res, show

    def detect_video(self, video_path):
        capture = cv2.VideoCapture(video_path)
        while True:
            _, image = capture.read()
            if image is None:
                break
            cv2.imwrite('./save.png', image)
            img_path = './save.png'
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            plt.imshow(img)
            img = self.data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            with torch.no_grad():
                output = torch.squeeze(self.model(img.to(self.device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            plt.title(print_res)
            plt.savefig('./save.png', format='png')
            show = cv2.imread('./save.png')
            yield print_res, show



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # ...

    def retranslateUi(self, MainWindow):
        # ...

    def openfile2(self):
        # ...

    def handleCalc4(self):
        # ...

    def openfile(self):
        # ...

    def handleCalc3(self):
        # ...

    def printf(self, text):
        # ...

    def showimg(self, img):
        # ...

    def click_1(self):
        # ...


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
