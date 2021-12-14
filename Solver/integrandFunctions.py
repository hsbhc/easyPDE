from Solver.question import Question


class IntegrandFunction():
    def __init__(self,question:Question,trial,test):
        self.question = question
        self.rs_left=question.rs_left
        self.rs_right= question.rs_right
        self.trial=trial
        self.test=test
        self.a=0
        self.b=0
        self.alpha=0
        self.beta=0

    def set(self,a,b,alpha,beta):
        self.a=a
        self.b=b
        self.alpha=alpha
        self.beta=beta

    def A_F(self,x):
        trial=self.trial(x, (self.a, self.b), index=self.alpha, derivative=self.rs_left[0])
        test=self.test(x, (self.a, self.b), index=self.beta, derivative=self.rs_left[1])
        f = self.question.C(x) * trial*test
        return f

    def B_F(self,x):
        test = self.test(x, (self.a, self.b), index=self.beta, derivative=self.rs_right[0])
        f = self.question.F(x)  * test
        return f