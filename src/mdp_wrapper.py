import mdp

class ListInputOutputNode(mdp.Node):
    
    def __init__(self, *args, **kwargs):
        super(ListInputOutputNode, self).__init__(*args, **kwargs)
    
    def execute(self,x):
        output = None
        if type(x) == type([]):
            output = []
            for item in x:
                output += self._execute(item)
        else:
            output = self._execute(x)
        return output

class ListInputNode(mdp.Node):    
    def __init__(self, *args, **kwargs):
        super(ListInputNode, self).__init__(*args, **kwargs)
    
    def execute(self,x):
        """ we are taking a list and turning it into an array, so let's assume it's the right type """
        output = None
        if type(x) == type([]):
            output = self._execute(x)
        else:
            output = self._execute([x])
        return output
