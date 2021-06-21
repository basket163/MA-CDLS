

class clsUser:
    """User Class"""
    userCount = 0

    def __init__(self,userId,userType,userInCell,profileInServ,requestUnit):
        clsUser.userCount += 1
        self.userId = userId
        self.userType = userType
        self.userInCell = userInCell
        self.profileInServ = profileInServ
        self.requestUnit = requestUnit

    def showUserCount(self):
        print("user count: {}".format(clsUser.userCount))

    def showUser(self):
        print("userId {}, inCell {}, inServ {}".format(self.userId,self.userInCell,self.profileInServ))

    def getUserRequest(self):
        return self.requestUnit

class clsServ:
    """Server class"""
    def __init__(self,serverId,servInCell,capa,haveUserNum,haveUserList):
        self.serverId = serverId
        self.servInCell = servInCell
        self.capa = capa
        self.haveUserNum = haveUserNum
        self.haveUserList = haveUserList

    def showServ(self):
        print("servId {}, inCell {}, haveUser {}".format(self.serverId,self.servInCell,self.haveUserNum))

class clsCell:
    """Cell Class"""
    def __init__(self,matrixCell):
        self.matrixCell = matrixCell

    def getCellDistance(self, cellx, celly):
        row = cellx - 1
        col = celly - 1
        value = self.matrixCell[row,col]
        return value

class clsSlot:
    def __init__(self,dfMove):
        self.dfMove = dfMove
        self.userList = []
        self.servList = []

    

    
    
