import operator
import random, numpy, math
from deap import creator, base, tools, gp, algorithms
from sklearn.metrics import r2_score as r2
import itertools
import csv
import data_man

'''
DEFINE PRIMIITIVE SET
'''
def protectedDiv(left, right):
    try:
        return operator.truediv(left, right)
    except ZeroDivisionError:
        return 1


#pset = gp.PrimitiveSet("MAIN", data_man.dummydiamond_xs.shape[1])
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, data_man.cancer_xs.shape[1]), bool)
pset.addPrimitive(numpy.add, [float, float], float)
pset.addPrimitive(numpy.subtract, [float, float], float)
pset.addPrimitive(numpy.multiply, [float, float], float)
#pset.addPrimitive(protectedDiv, [float, float], float)

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# logic operators

# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

''' Give Args descriptive names
args = data_man.args

for i in range(len(args)):
    pset.renameArguments(args[i])

'''

'''
CREATE
'''
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

'''
DEFINE TOOLBOX
'''
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


'''
FITNESS FUNCTION: TODO
'''
def evalSymbReg(individual):


    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)


    '''
    TODO FIX THIS ISSUE P
    '''
    
    predicted = data_man.cancer_xs.apply(func, axis=1) 

    # Evaluate the error between predicted and actual
    error = sum(bool(predicted) - bool(data_man.diagnosis))
    return error / len(predicted)


'''
DEFINE TOOLBOX CONT.
'''

toolbox.register("evaluate", evalSymbReg) ## THIS CALLS THE EVALUATION FUNCTION
toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=2, fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


#TODO see if these decorators are neccessary with Double Tournament vs Single Tournament
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


'''
START EVOLUTION
'''
def run(popsize, gens, cxpb, mutpb):
    pop = toolbox.population(popsize)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, gens, stats=mstats, halloffame=hof, verbose=True)



'''
STATISTICAL MEASURES AND TOOLS
'''

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)



if(__name__ == "__main__"):
    run(100, 100, 0.5, 0.1)