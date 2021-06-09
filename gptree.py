import operator
import random, numpy, math
from deap.tools.support import Logbook
from deap import creator, base, tools, gp, algorithms
from sklearn.metrics import r2_score as r2
import itertools
import matplotlib.pyplot as plt
import networkx as nx
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
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protectedDiv, [float, float], float)

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
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
#pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

#pset.renameArguments(**data_man.args) #rename args


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

    random_classifier_accuracy = 0.5324699392

    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    sum_error = 0
    for datapoint in data_man.breastcancer.iterrows():

        # Evaluate the error between predicted and actual
        predicition = bool(func(*datapoint[1][1:31]))
        #print("prediction: {0}", predicition)
        actual = bool(datapoint[1][0])
        #print("actual: {0}", actual)
        if not(predicition) is actual:
            sum_error+=1

    accuracy = (data_man.cancer_xs.shape[0] - sum_error) / data_man.cancer_xs.shape[0]
    kappa = (accuracy - random_classifier_accuracy) / (1 - random_classifier_accuracy)
    return kappa, 

'''
DEFINE TOOLBOX CONT.
'''

toolbox.register("evaluate", evalSymbReg) ## THIS CALLS THE EVALUATION FUNCTION
#toolbox.register("select", tools.selTournament, tournsize=3, fit_attr="fitness") #single tournament selection  
toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.6, fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


'''
START EVOLUTION
'''
def run(popsize, gens, cxpb, mutpb):
    pop = toolbox.population(popsize)
    hof = tools.HallOfFame(5)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb, mutpb, gens, stats=mstats, halloffame=hof, verbose=True)

    summary_plots(hof, log)
    




def summary_plots(hof, log):
    print_hof(hof)
    graph_avg_fitness(log)
    graph_max_fitness(log)


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


'''
GRAPHING
'''
def print_hof(hof):
    for i in range(len(hof)):
        graph_ind(hof[i], i)

        #print("HOF" + str(i) + "fitness: " +  str(hof[i].fitness))


def graph_ind(ind, i):

    nodes, edges, labels = gp.graph(ind)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)

    plt.title("Hall of Fame #" + str(i) + ", fitness =" + str(ind.fitness))
    plt.show()

def graph_avg_fitness(log):
    gen = log.select('gen')
    avg = log.chapters['fitness'].select('avg')
    #make figure
    fig, ax = plt.subplots()
    ax.scatter(gen, avg)
    ax.set(xlabel = "# of Evalulations", ylabel = "Average Fitness of Gen")
    ax.grid()
    plt.show()


def graph_max_fitness(log):
    gen = log.select('gen')
    avg = log.chapters['fitness'].select('max')
    #make figure
    fig, ax = plt.subplots()
    ax.scatter(gen, avg)
    ax.set(xlabel = "# of Evalulations", ylabel = "Max Individual Fitness in Gen")
    ax.grid()
    plt.show()

if(__name__ == "__main__"):
    run(200, 500, 0.5, 0.08)