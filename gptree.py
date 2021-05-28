import operator
import random, numpy, math
from deap import creator, base, tools, gp, algorithms


'''
IMPORT DATASET TODO
'''


'''
DEFINE PRIMIITIVE SET
'''

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(operator.truediv, 2) #TODO prevent division by zero
pset.renameArguments(ARG0='x')



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
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    sqerrors = (func(x)*3 for x in points) #replace with adj r^2
    return math.fsum(sqerrors) / len(points),


'''
DEFINE TOOLBOX CONT.
'''
toolbox.register("evaluate", evalSymbReg, points=(1.0,)) #TODO points will refer to regressor variables from the dataset
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