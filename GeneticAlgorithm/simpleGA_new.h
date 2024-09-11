/*--------------------------------------------------------------------------------*
 * A Simple Genetic Algorithm (SGA), based on the algorithm presented in the book: *
 * Michalewicz, Z., "Genetic Algorithms + Data Structures = Evolution Programs",  *
 * Springer-Verlag, 3rd edition, 1996.                                            *
 *--------------------------------------------------------------------------------*/

/*--------------- SGA.H  Header File ---------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define frand_ab() 	(2*((double) rand() / RAND_MAX)-1)
#define frand()    	((double) rand()/RAND_MAX)
#define sgmax(a,b)	(a>b ? a : b)
#define sgmin(a,b)	(a<b ? a : b)
/*--------------------------------------------------------------------------------
Change the folowing parameters to match your needs */
#define NPOINTS		 12 //25					/* The number of points (villages/cities) */
#define CITIES_FILE	"Halkidiki_12.txt"	 // "Halkidiki_25.txt"
#define RESULTS_FILE "results_12.txt"	// "results_25.txt"
#define POPSIZE		50//1000				/* Population size */
#define PXOVER		0.5//0.7					/* Probability of crossover */
#define PMUTATION	0.1//0.05				/* Probability of mutation */
#define MAXGENS		1000//100					/* Maximum number of generations */
#define DIMENSION	2					/* The dimension of the space (default Euclidean space */
#define DISLPAYFREQ	50//50					/* Print to screen each DISLPAYFREQUENCY generations */
#define EXPERIMENTS	10					/* Number of experiments */
#define TRUE		1
#define FALSE		0
/*---------------------------------------------------------------------------------*/
int Generation;							/* Current Generation number */
int Best;								/* The Best genotype in the population */
FILE *results;							/* A log file to write results to */
float Cities_data[NPOINTS][DIMENSION];	/* The table that holds the problem data */

struct genotype                 		/* Each genotype is a member of the population */
{                               
  double Gene[DIMENSION],					/* A string of co-ordinates that makes a genotype */
  Fitness,								/* The genotype's fitness */
  Upper[DIMENSION],						/* The genotype's upper bound */
  Lower[DIMENSION],						/* The genotype's lower bound */
  RFitness,								/* The relative fitness */
  CFitness,								/* The comulative fitness */
  Selection_Prob,						/* The probability of a selection */
  Cumulative_Prob;						/* The cumulative probability */
  int  Survivor,						/* Flag for selection procedure */
  Mate,									/* Flag for Crossover procedure */
  Mutate;								/* Flag for Mutation procedure */
};
struct genotype Population[POPSIZE+1];	/* The population of genotypes */
struct genotype Best_Individual;

/* general functions */
double RandVal(double,double);
void LoadCities(void);
double EuclideanDistance(double*, float*, const int);

/* GA function prototypes */
void initialize(void);
void evaluate(void);
void copy_genotypes(struct genotype*, struct genotype*);
void copy_population(struct genotype old_pop[POPSIZE+1], struct genotype new_pop[POPSIZE+1]);
void select(void);
void crossover(void);
void mutate(void);
void report(void);


