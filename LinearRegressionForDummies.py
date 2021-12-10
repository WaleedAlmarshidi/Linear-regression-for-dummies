from math import sqrt
import math
import numpy as np
import logging
from numpy import ma
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import scipy.stats as st
import pandas as pd
import copy
import time
from UserUi import MainPage
from statsmodels.sandbox.stats import runs
from statsmodels.graphics.gofplots import qqplot
from matplotlib import colors, pyplot
from scipy.stats import anderson
import seaborn as sns
pyplot.style.use('seaborn')
sns.set_theme(style="dark")


class Coefficint:
    def __init__(self, value, name):
        self.Name = name
        self.Value = value
class RepeatedXValues:
    def __init__(self, x, CorrespondingYs):
        self.CorrespondingYs = CorrespondingYs
        self.XValue = x
class ModelUnderTest:
    def __init__(self, model, addedPredictorVariableIndex, testvalue = None):
        self.Model = model
        self.AddedPredictorVariableIndex = addedPredictorVariableIndex
        self.TestValue = testvalue

class Plots:
    def ApplyPlotStyle(self, title = '', XLabel = '', YLabel = ''):
        BaseColor = 'white'
        pyplot.figure(facecolor='black')
        ax = pyplot.axes()
        ax.set_facecolor("black")
        ax.xaxis.label.set_color(BaseColor)
        ax.yaxis.label.set_color(BaseColor)
        ax.tick_params(axis='x', colors= BaseColor)
        ax.tick_params(axis='y', colors= BaseColor)
        ax.spines['left'].set_color(BaseColor)
        ax.spines['bottom'].set_color(BaseColor)
        pyplot.title(title, color = 'white', fontweight='bold')
        pyplot.xlabel(XLabel)
        pyplot.ylabel(YLabel)
        ax.grid(False)

    def PlotXY(self):
        if self.p == 2:
            self.ApplyPlotStyle(title='X vs Y', XLabel= 'X', YLabel= 'Y')
            pyplot.scatter(self.XList[0], self.y, color = 'violet')
            # pyplot.title('X vs Y')
            self.UI.AddGraphToModelBuildingSection(pyplot.gcf())
            pyplot.close()

    def PlotEYHat(self):
        self.ApplyPlotStyle()
        sns.residplot(
            self.PredictionVector(self.X), 
            self.Resids,
            lowess=True,
            color= 'violet'
            )
        pyplot.ylabel('Error')
        pyplot.xlabel('E(Y)')
        pyplot.title('Plot to check for\nError trending / Heteroscedacisity', color= 'white', fontweight='bold')
        self.UI.AddGraphToErrorTestsSection(pyplot.gcf())
        pyplot.close()

    def QQPlot(self):
        self.ApplyPlotStyle(title= 'QQ plot', XLabel= 'Theoretical normal value', YLabel= 'Acutal value')
        qqplot(self.Resids, line='s', markerfacecolor='violet')
        self.UI.AddGraphToErrorTestsSection(pyplot.gcf())
        pyplot.close()
    
    def PlotVIFs(self, X, Y):
        pyplot.bar(X, Y)
        self.UI.AddGraphToModelFilteringSection(pyplot.gcf())
        pyplot.close()

    def PlotEOrder(self):
        self.ApplyPlotStyle(title= 'Error vs Order', XLabel= 'Order', YLabel= 'Error')
        pyplot.scatter([i for i in range(len(self.Resids))], self.Resids)
        self.UI.AddGraphToErrorTestsSection(pyplot.gcf())
        pyplot.close()

class Test(Plots):
    def BadNewsTextDecoration(self, msg):
        return f'[color=ff2e2e]{msg}[/color]'
  
    def GoodNewsTextDecoration(self, msg):
        return f'[color=21ff4e]{msg}[/color]'
  
    def CantOperateTestTextDecoration(self):
        pass
    
    def HeaderTextDecoration(self, msg):
        return f'[size=40][b][i]{msg}[/i][/b][/size]'

    def GetRidOfVarianceInflationFactor(self, Threshold = 5):
        if self.p < 3:
            return
        Logs = self.HeaderTextDecoration('Multicollinearity detection and filtering:')
        ModelsUnderTest = []
        for i in range(len(self.XList)):
            ModelsUnderTest.append(
                ModelUnderTest(
                    LinearRegression(self.XList[i], *[self.XList[q] for q in range(len(self.XList)) if q != i])
                        , i
                        )
                    )
            ModelsUnderTest[-1].TestValue = 1/(1 - ModelsUnderTest[-1].Model.RS())

        VIFs = [i.TestValue for i in ModelsUnderTest]
        MaximumVIF = max(VIFs)
        
        
        if MaximumVIF < 5:
            Logs += self.GoodNewsTextDecoration('\nNo significant Multicollinearity exists in the model')
        else:
            VIFsGreaterThan5 = [i for i in VIFs if i > Threshold]
            MulticollinearityModels = [i for i in ModelsUnderTest if i.TestValue > Threshold]
            for i in MulticollinearityModels:
                pass
            LeastInfluentialVariable = min(VIFsGreaterThan5)
            IndexOfLeastInfluentialVariable = VIFsGreaterThan5.index(LeastInfluentialVariable)
            Logs += "Variance inflation factor found at {}".format(0)
            
        self.UI.AddToModelSelectionLogs(Logs)
        logging.critical("Variable removeed !!")
        return ['X' + str(i.AddedPredictorVariableIndex + 1) for i in ModelsUnderTest], VIFs

    def RunsTestForRandomness(self, alpha = 0.05):
        runs, n1, n2 = 0, 0, 0
        l = self.Resids
        l_median = median(l)
        # Checking for start of new run
        for i in range(len(l)):
            # no. of runs
            if (l[i] >= l_median and l[i-1] < l_median) or \
                    (l[i] < l_median and l[i-1] >= l_median):
                runs += 1  
            # no. of positive values
            if(l[i]) >= l_median:
                n1 += 1   
            # no. of negative values
            else:
                n2 += 1   
    
        runs_exp = ((2*n1*n2)/(n1+n2))+1
        stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
                        (((n1+n2)**2)*(n1+n2-1)))
    
        z = (runs-runs_exp)/stan_dev
    
        P_value = st.norm.sf(abs(z))*2
        FinalDecesion = self.HeaderTextDecoration("\nUsing Runs-test-for-randomness")
        FinalDecesion += '\nH0: the errors are random'
        FinalDecesion += '\nH1: the errors are not random'

        if P_value > alpha:
            FinalDecesion += self.GoodNewsTextDecoration("\nWe expect the errors to follow a random pattern")
        else:
            FinalDecesion += self.BadNewsTextDecoration("\nThe errors DO NOT follow random pattern[/color]\nthus you may consider more explanatory vaiables to enter")
        FinalDecesion += "\nRuns = {}\nP value = {}".format(runs, round(P_value, 4))
        self.UI.AddToErrorTestsLogs(FinalDecesion)

    def BartlettsTestForHomogeneity(self):
        SplittedResids = np.array_split(self.Resids, 2) 
        Group1 = SplittedResids[0]
        Group2 = SplittedResids[1]
        _, P_value = st.bartlett(Group1, Group2)
        FinalDecesion = self.HeaderTextDecoration('Using Bartletts test for homogeneity :')
        if P_value > 0.05:
            FinalDecesion += self.GoodNewsTextDecoration('\nWe expect that the varaince of the error is steady,\nWhich means that the model is mostly appropriate')
        else:
            FinalDecesion += self.BadNewsTextDecoration('We cant conclude that the varaince of the error is steady,')
            FinalDecesion += '\nWhich means that you may consider to fit more variables'
        FinalDecesion += f'\nP value : {round(P_value, 4)}'
        self.UI.AddToErrorTestsLogs(FinalDecesion)

    def LackOfFitTest(self, alpha = 0.05):
        UniqueXs = []
        XYPairs = []
        C = 0
        Logs = self.HeaderTextDecoration("Lack of fit test:")
        for Xi in self.XList:
            for xi in Xi:
                if xi not in UniqueXs:
                    FirstUniqueXsIndex = Xi.index(xi)
                    # logging.critical(xi)
                    UniqueXs.append(xi)
                    XYPairs.append(RepeatedXValues(xi, [self.y[FirstUniqueXsIndex]]))
                    C += 1
                    for q in range(FirstUniqueXsIndex + 1, len(Xi)):
                        if Xi[q] == xi:
                            XYPairs[-1].CorrespondingYs.append(self.y[q])
        if C == 0:
            self.UI.AddToModelSelectionLogs('\nCant operate LOF test\ndue to lack of repetition in the x values')
            return

        sspe = 0
        for YGroup in XYPairs:
            for element in YGroup.CorrespondingYs:
                sspe += pow(mean(YGroup.CorrespondingYs) - element, 2)
                
        sslf = self.SSE() - sspe
        LfDof = C - self.p
        PeDof = self.n - C
        mslf = sslf/(LfDof)
        mspe = sspe/(PeDof)
        FNote = mslf/mspe
        Fcrit = st.f.ppf(1 - alpha, LfDof, PeDof)
        if FNote > Fcrit:
            Logs += self.BadNewsTextDecoration("\nLack of fit has been detected,")
            Logs += "\nData transformation might be necessary"
        else:
            Logs += self.GoodNewsTextDecoration("\nNo significant lack of fit exist in the model\nwhich means that the model is mostly appropriat")
        Logs += "\nF* = {}\nF_alpha = {}".format(round(FNote, 3), round(Fcrit, 3))
        self.UI.AddToModelSelectionLogs(Logs)
        return mslf/mspe

    def AndersonDarlingForNormality(self, alpha = 0.05):
        H0 = '\nH0 : The errors follow a nromal dis'
        H1 = '\nH1 : The errors does not follow a nromal dis'
        
        TestResult = anderson(self.Resids)
        AD = TestResult.statistic
        CV = TestResult.critical_values[np.where(TestResult.significance_level == alpha*100)[0][0]]
        FinalDecesion = self.HeaderTextDecoration('Using Anderson-darling test for normality')
        FinalDecesion += H0
        FinalDecesion += H1
        if AD > CV:
            FinalDecesion += self.BadNewsTextDecoration('\nWe faild to confirm the normality of the erros')
            FinalDecesion+= '\nconcluding that the model is mostly not appropriat'
        else:
            FinalDecesion += self.GoodNewsTextDecoration('\nWe conclude that the errors follow a normal distribution')
        P_Value = 0
        AD = AD*(1 + (.75/50) + 2.25/(50**2))
        if AD >= .6:
            P_Value = math.exp(1.2937 - 5.709*AD - .0186*(AD**2))
        elif AD >=.34:
            P_Value = math.exp(.9177 - 4.279*AD - 1.38*(AD**2))
        elif AD >.2:
            P_Value = 1 - math.exp(-8.318 + 42.796*AD - 59.938*(AD**2))
        else:
            P_Value = 1 - math.exp(-13.436 + 101.14*AD - 223.73*(AD**2))
        FinalDecesion += f'\nAD = {round(AD, 4)}' + f'\nP value = {round(P_Value, 4)}'
        self.UI.AddToErrorTestsLogs(FinalDecesion)

    def FilterOutliers(self, threshold = 3):
        Logs = self.HeaderTextDecoration('Outlire detection and filtering: ')
        Counter = 0
        for i in range(self.n - Counter - 1):
            Z = self.Normalise(self.Resids[i])
            if abs(Z) > threshold:
                Counter += 1
                self.Resids = np.delete(self.Resids, i)
                Logs += self.BadNewsTextDecoration(f'\nOutlire found at : {i}')
                Logs += f'with {Z} absolute deviation.'
                # self.y = np.delete(self.y, i)
                # self.X = np.delete(self.X, i, axis=0)
                # logging.critical(self.X)
                # for r in self.XList:
                #     r.pop(i)

        if Counter == 0:
            Logs += self.GoodNewsTextDecoration('\nNo outlire found.')
        self.UI.AddToModelSelectionLogs(Logs)

class LinearRegression (Test):
    def __init__(self, *args):
        self.BaseXs = None
        self.logs = ""
        if len(args) == 1 and isinstance(args[0], str):
            self.InitialiseByFilePath(args[0])
        else:
            self.InitialiseByDirectPythonTypes(args[0], args[1:])
        
        # print("Xlist : ")
        # logging.critical(self.XList)
        # logging.critical(self.X)
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        # logging.info(self.X)
        # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        # logging.info(self.y)

        self.BVector = self.Fit()
        self.sse = round(self.SSE(), 4)
        self.sst = round(self.SST(), 4)
        self.Resids = self.Residuals()
        self.EDoF = self.n - self.p
        self.CorrelationMatrix = np.dot(self.MSE(), np.linalg.inv(np.dot(np.transpose(self.X), self.X)))
        # logging.critical(self.CorrelationMatrix)
        self.PredictorVarsInfo = self.BInfo()
        self.rs = round(self.RS(), 4)
        # logging.critical(self.FindPredictorVarIndex(self.XList[1]))

    def Normalise(self, data):
        return data/sqrt(self.MSE())

    def ShowRS(self):
        Logs = f'R squared : {self.rs}'
        Logs += f'\nwhich means that {self.rs*100}% of the variation in Y is explained by all of your linear component'
        Logs += '\nthe sensivity of the study/expirement determins The good percentage threshold'
        self.UI.AddToModelBuildingLogs(Logs)
    
    def ShowSSE(self):
        Logs = f'SSE : {self.sse}'
        self.UI.AddToModelBuildingLogs(Logs)
    
    def ShowCoeffInformation(self):
        self.UI.AddToModelBuildingLogs(self.Summary())

    def Auto(self):
        self.UI = MainPage()
        self.UI.AddToModelBuildingLogs(self.HeaderTextDecoration('Beggining'))
        self.ShowCoeffInformation()
        self.UI.AddToModelBuildingLogs(self.HeaderTextDecoration('Summary stats you may need :\n'))
        self.ShowRS()
        self.ForwaredSelection()
        self.PlotEYHat()
        self.PlotXY()
        self.FilterOutliers()
        VIFs = self.GetRidOfVarianceInflationFactor()
        self.PlotVIFs(VIFs[0], VIFs[1])
        self.LackOfFitTest()
        self.AndersonDarlingForNormality()
        self.QQPlot()
        self.RunsTestForRandomness()
        self.PlotEOrder()
        self.BartlettsTestForHomogeneity()
        self.UI.Run(self.UI)
    
    def UserLogs(self):
        BaseMessage = "{}\n SSE: {}\n SSR: {}\n RS: {}\n Warnings and important messages : {}"
        return BaseMessage.format(self.Summary(), self.sse, self.sst - self.sse, self.rs, self.logs)
        
    def AddToUserLogs(self, msg):
        self.logs += str(msg)

    def InitialiseByFilePath(self, FilePath):
        df = pd.read_csv(FilePath)
        data = df.values
        self.y = data[:, -1]
        self.X = np.array(data[:, :-1])
        self.n = df.shape[0]
        col = [list(data[:, i]) for i in range(df.shape[1] - 1)]
        # for i in range(df.shape[1]):
        #     print(list(data[:, i]))
        self.XList = col
        self.X = np.append(np.ones((self.n, 1)), self.X, 1)
        self.p = df.shape[1]
        # logging.info(self.p)

    def InitialiseByDirectPythonTypes(self, y, *args) :
        self.y = np.array(y)
        self.n = len(y)
        self.XList = args[0]
        self.X = [np.ones(self.n)]
        for i in args:
            if isinstance(i, tuple) or isinstance(i, tuple): 
                for c in i:
                    self.X.append(c)
                continue
            self.X.append(i)
        self.X = np.transpose(self.X)
        if args[0] != ():
            self.p = len(args) + 1
        else:
            self.p = 1
    
    def FindPredictorVarIndex(self, X):
        # logging.critical(BaseExplanatoryVars)
        BaseExplanatoryVars = self.BaseXs if self.BaseXs != None else self.XList
        for i in range(len(BaseExplanatoryVars)):
            if all(x in X for x in BaseExplanatoryVars[i]):
                # logging.error(BaseExplanatoryVars[i])
                # logging.critical(i)
                return i
        return None

    def PlotData(self):
        pass

    def BInfoRelativeToBaseModel(self):
        return self.PredictorVarsInfo.index()

    def BInfo(self):
        self.PredictorVarsInfo = []
        for i in range(len(self.BVector)):
            if i != 0:
                
                self.PredictorVarsInfo.append(
                    Coefficint(
                        round(self.BVector[i], 4), 
                        ' X' + str(self.FindPredictorVarIndex(self.XList[i - 1]) + 1)
                        )
                    )
            else:
                self.PredictorVarsInfo.append(
                    Coefficint(
                        round(self.BVector[i], 4), 
                        ' X' + str(0)
                        )
                    )
        return self.PredictorVarsInfo

    def AddPredictorVariable(self, newX, BaseXs):
        # print('+=======================================')
        #print(newX)
        # logging.info(self.X)
        self.XList += (newX,)
        #logging.critical(self.XList)
        NewModel = LinearRegression(self.y, *self.XList)
        NewModel.BaseXs = BaseXs
        # logging.error(BaseXs)
        return NewModel

    def RemovePredictorVariable(self, X):
        self.XList = self.XList[:X]
        return LinearRegression(self.y, *self.XList)
        
    def ForwaredSelection(self, alpha = 0.05):
        BestModel = LinearRegression(self.y)
        PrevAcceptedPredictorsIndeces = []
        logging.info("FS")
        UserLogs = self.HeaderTextDecoration('Model selection began')
        for c in range(self.p - 1):
            VirtualBestModel = copy.copy(BestModel)
            TestedModels = []
            for i in range(self.p - 1): # Fitting all possible models 
                if i not in PrevAcceptedPredictorsIndeces:
                    TestedModels.append(
                        ModelUnderTest(VirtualBestModel.AddPredictorVariable(self.XList[i], self.XList), i)
                        )
                    VirtualBestModel = copy.copy(BestModel)
            
            UserLogs += f'\nComparing {len(TestedModels)} models'
            UserLogs += '\n...'
            UserLogs += '\n...'
            # Calculating partial f test for each model
            TestedModelPartialFTest = [i.Model.Ftest() for i in TestedModels]
            HeighestFTest = max(TestedModelPartialFTest)
            BestModelIndex = TestedModelPartialFTest.index(HeighestFTest)
            if HeighestFTest > st.f.ppf(1 - alpha, 1, self.EDoF):
                BestModel = TestedModels[BestModelIndex].Model
                PrevAcceptedPredictorsIndeces.append(TestedModels[BestModelIndex].AddedPredictorVariableIndex)
                UserLogs += self.GoodNewsTextDecoration(f"\nX{self.FindPredictorVarIndex(BestModel.XList[-1]) + 1} added to the model\nwith {round(HeighestFTest, 4)} partial F test statistic")
                UserLogs += BestModel.Summary()
            else:
                UserLogs += "\nApproved model :\n"
                UserLogs += BestModel.Summary()
                self.UI.AddToModelSelectionLogs(UserLogs)
                return BestModel

        UserLogs += BestModel.Summary()
        UserLogs += self.BadNewsTextDecoration("Forwared selection algorithm selected the full model")
        self.UI.AddToModelSelectionLogs(UserLogs)
        BestModel.Summary()        
        return BestModel
  
    def Summary(self):
        message = self.HeaderTextDecoration("\nCoefficints information :\n")
        # self.BInfo()
        # self.PredictorVarsInfo = self.PredictorVarsInfo
        for i in self.PredictorVarsInfo:
            message += i.Name + ': ' + str(i.Value) + '\n'
        # logging.info(message)
        return message
    
    def Ftest(self):
        return (self.getReducedModel().SSE() - self.SSE())/ self.MSE()

    def BTest(self, coeindex = 1):
        return self.BVector[coeindex] / self.CorrelationMatrix[coeindex, coeindex]

    def MSE(self):
        return self.sse / self.EDoF

    def Fit(self):
        XT = np.transpose(self.X)
        p1 = np.dot(XT, self.X)
        p2 = np.dot(XT, self.y)
        B = np.dot(np.linalg.inv(p1), p2)
        return B

    def Residuals(self):
        predictions = self.PredictionVector(self.X)
        e = np.subtract(self.y, predictions)
        return e
    
    def SSE(self):
        E = self.Residuals()
        SSE = np.dot(E, np.transpose(E))
        # logging.critical(SSE)
        return SSE

    def SST(self):
        ssy = np.dot(np.transpose(self.y), np.dot(np.identity(self.n) - (1 / self.n)*np.ones(self.n), self.y))
        return ssy

    def getReducedModel(self):
        # print('----------------------------' )
        # logging.info(x)
        x = self.XList[: len(self.XList) - 1]
        ReducedModel = LinearRegression(self.y, *x)
        return ReducedModel

    def RS(self):
        return 1 - self.SSE() / self.SST()

    def Predict(self, x):
        return self.BVector[0] + self.BVector[1]*x

    def PredictionVector(self, X):
        return np.dot(X, self.BVector)

