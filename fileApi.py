import cv2
import re

class fileAPI(object):
    def __init__(self):
        self.textData = {}
        self.imageData = {}
        self.results = {}

    def getImage(self, fileName):
        if fileName == None:
            return
        data = None
        try:
            data = self.imageData[fileName]
        except:
            raise 'Invalid file name.'
        return data

    def getTextFile(self, fileName):
        if fileName == None:
            return
        data = None
        try:
            data = self.textData[fileName]
        except:
            raise 'Invalid file name.'
        return data

    def getResult(self, fileName):
        if fileName == None:
            return
        data = None
        try:
            data = self.results[fileName]
        except:
            raise 'Invalid file name.'
        return data

    def addResult(self, resultName, data):
        if resultName == None:
            return
        try:
            self.results[resultName] = data
        except:
            raise 'Couldnt add result.'
        return

    def saveResults(self, path):
        if path == None:
            return
        for key, value in self.results.iteritems():
            file = open(str(key), 'r')
            file.write(value)
            file.close()

    def compareTextFile(self, resultName, orginalFileName):
        resultData = self.results[resultName]
        orginalFile = self.textData[orginalFileName]
        
        minLen = min(len(resultData), len(orginalFile))
        maxLen = max(len(resultData), len(orginalFile))
        diffData = ''
        for i in range(0, minLen):
            if resultData[i] != orginalFile[i]:
                diffData += 'x'
            else:
                diffData += resultData[i]

        diffDataLen = maxLen - minLen

        diffData += 'x'*diffDataLen

        return diffData
            
    def readFiles(self, paths):
        fileNames = []
        if paths == None:
            return
        fileNamePattern = r'.*\\(.*\..*)'
        for path in paths:
            match = re.match(fileNamePattern, path)
            if match:
                fileName = match.group(1)
                if path == None:
                    continue
                fileNames.append(fileName)
                if path[-3:].lower() == 'png':
                    if fileName not in self.imageData:
                        self.imageData[fileName] = cv2.imread(path)
                    else:
                        raise 'File exists in collection'
                elif path[-3:].lower() == 'txt':
                    if fileName not in self.textData:
                        file = open(path, 'r')
                        data = file.read()
                        file.close()
                        self.textData[fileName] = data
                    else:
                        'File exists in collection'
                else:
                    print(fileName)
                    raise 'Invalid file type'
        return fileNames

if __name__ == "__main__":
    def compareIMG(img1, img2):
        if img1.shape == img2.shape:
            difference = cv2.subtract(img1, img2)
            if cv2.countNonZero(difference) == 0:
                return True

        return False

    def fileApiUnitTests():
        api = fileAPI()

        assert api.textData == {}
        assert api.imageData == {}
        assert api.results == {}

        api.readFiles(['./data/macbeth.txt', './data/test.txt', './data/random.png', './data/random1.png'])
        file2 = open('./data/macbethChanged.txt', 'r')
        data = file2.read()
        file2.close()
        api.addResult('macbethChanged.txt', data)

        file = open('./data/macbeth.txt', 'r')
        data = file.read()
        file.close()
        assert api.getTextFile('macbeth.txt') == data

        assert api.compareTextFile('macbethChanged.txt', 'macbeth.txt')

        file = open('./data/test.txt', 'r')
        data = file.read()
        file.close()
        assert api.getTextFile('test.txt') == data

        img = cv2.imread('./data/random.png', 0)
        assert compareIMG(api.getImage('random.png'), img)

        img = cv2.imread('./data/random1.png', 0)
        assert compareIMG(api.getImage('random1.png'), img)


        print('Pass')

    fileApiUnitTests()

