import operator
import random, numpy, math
from deap import creator, base, tools, gp, algorithms
from sklearn.metrics import r2_score as r2
import itertools

import data_man

'''
DEFINE PRIMIITIVE SET
'''
def protectedDiv(left, right):
    try:
        return numpy.true_divide(left, right)
    except ZeroDivisionError:
        return left


#pset = gp.PrimitiveSet("MAIN", data_man.dummydiamond_xs.shape[1])
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, len(data_man.points)), float)
pset.addPrimitive(numpy.add, 2)
pset.addPrimitive(numpy.subtract, 2)
pset.addPrimitive(numpy.multiply, 2)
pset.addPrimitive(protectedDiv, 2)



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
def evalSymbReg(individual): #, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    errors = (func(data_man.dummydiamond_xs) - data_man.diamond_y_price)# for x in points) #replace with adj r^2

    result = sum(float(func(*diamond[1:])) - float(diamond[0]) for diamond in data_man.dummydiamonds)
    return result #math.fsum(errors) / len(points),


'''
DEFINE TOOLBOX CONT.
'''
#print(data_man.diamond_xs)
#print(data_man.diamond_xs.to_numpy)
toolbox.register("evaluate", evalSymbReg) #, points=data_man.points) #TODO points will refer to regressor variables from the dataset
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