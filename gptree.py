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
FITNESS FUNCTION:
'''
def evalSymbReg(individual):


    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    sum_error = 0

    '''
    ERROR IS HERE

    I am very unfamiliar with python/pandas/DEAP so this has been quite challengenging to debug and simplify. line 78 compiles an indivual GP tree from the pset defined in line 20. 
    The tree is supposed to take 28 arguements, one for each regressor (x) variable in the dataset. I then want to loop through each  datapoint in the dataset and see if the predicted
    classification matches the labeled correct classification (located in column 0). The issue that occurs is that I cannot split the dataframe columns from each datapoint correctly,
    and get the error that the number of arguements do not match what func requires. Ive tried importing data both with the csv module and with pandas. I have tried to apply 'func' to
    the columns in multiple ways (using pandas.iterrows() line 93, pandas.apply() line 101, as well as manually looping though each 2-D index line 103. I have also attempted using different datasets. I've been
    struggling with this for close to 20 hours at this point and I am not sure how to proceed....
   
   '''
    for datapoint in data_man.breastcancer.iterrows():
        predicition = bool(func(*datapoint[1:31]))
        actual = datapoint[0]
        error = predicition is actual

        sum_error+=error


    #predicted = data_man.cancer_xs.apply(func, axis=1) 
    # Evaluate the error between predicted and actual
    #sum_error = sum(bool(func(*datapoint[1:31])) is datapoint[0] for datapoint in data_man.breastcancer) # This line returns an an error that I am missing 21 positional arguements when calling 'func'

    return sum_error, #/ len(predicted)

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