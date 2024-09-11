#include "simpleGA_new.h"	/* include the sga.h header file */

/*------------------------------------------------------*/
/*              RandVal                                 */
/* returns a pseudorandom value between the supplied    */
/* low and high arguments.                              */
/*                                                      */
double RandVal (double low, double high)
{
	return ((((double) rand())/RAND_MAX)*(high-low))+low;
}
/*------------------------------------------------------*/
/*				  LoadCities		                    */
/* Loads the cities coordinates from a file 		    */
/*                                                      */
void LoadCities(void)
{
	FILE *finput;
	int i,j;

	if ((finput=fopen(CITIES_FILE, "r")) == NULL)
	{
		printf("I can't open %s\n", CITIES_FILE);
		exit(1);
	}
	for (i=0; i<NPOINTS; i++)
	{
		for (j=0; j<DIMENSION; j++)
		{
			fscanf(finput,"%f", &Cities_data[i][j]);
			printf("%f ", Cities_data[i][j]);
		}
		printf("\n");
	}
	fclose(finput);
}
/*------------------------------------------------------*/
/* 				EuclideanDistance		                */
/* Calculate the Euclidean Distance of two points       */
/* in a space with dimension SpaceDim                   */
/*                                                      */
double EuclideanDistance(double *point1, float *point2, const int SpaceDim)
{
	double distance = 0;
    int i;
    
	for (i=0; i<SpaceDim; i++) distance += pow((point1[i]-point2[i]),2);
	distance = pow(distance,0.5);
	return distance;
}
/*------------------------------------------------------*/
/*              INITIALIZE                              */
/* create the initial population					    */
/*										                */
void initialize(void)
{
	LoadCities();
    int i,j;
    
	double lbound[DIMENSION],ubound[DIMENSION];
	for (j=0; j<DIMENSION; j++)
	{
		lbound[j] = Cities_data[0][j];
		ubound[j] = Cities_data[0][j];
	}

	/* Find the lower and upper bounds for each co-ordinate */
	/* using the original cities coordinates                */
	for (i=1; i<NPOINTS; i++)
	  for (j=0; j<DIMENSION; j++)
	  {
		lbound[j] = sgmin(lbound[j], Cities_data[i][j]);
		ubound[j] = sgmax(ubound[j], Cities_data[i][j]);
	  }

	/* set the bounds for each gene in the population */
	for (i=0; i<=POPSIZE; i++)
	  for (j=0; j<DIMENSION; j++)
	  {
		Population[i].Lower[j]=lbound[j];
		Population[i].Upper[j]=ubound[j];
	  }

	/* Using the lower and upper bounds, 		         	*/
	/* assign a random value to each gene of each genotype 	*/
	for (i=0; i<=POPSIZE; i++)
	  for (j=0; j<DIMENSION; j++)
	    Population[i].Gene[j] = RandVal(Population[i].Lower[j], Population[i].Upper[j]);
}
/*------------------------------------------------------*/
/*              EVALUATE                                */
/* Compute the performance of each member              */
/*	                     								*/
void evaluate(void)
{
	Best=0;
    int i,j,mem;
	/* calculate fitness */
	for (mem=0; mem<POPSIZE; mem++)
	{
		Population[mem].Fitness = 0;
		for (i=0; i<NPOINTS; i++)
		  Population[mem].Fitness += EuclideanDistance(Population[mem].Gene, Cities_data[i], (const int)DIMENSION);
		Population[mem].Fitness = 1.0/(Population[mem].Fitness);

		/* Keep track of the best member of the population  */
		/* Note that the last member of the population holds*/
		/* a copy of the best member.                       */
		if (Population[mem].Fitness > Population[POPSIZE].Fitness)
		{
			Best=mem;
			Population[POPSIZE].Fitness=Population[mem].Fitness;
			for (j=0; j<DIMENSION; j++)
			  Population[POPSIZE].Gene[j]=Population[mem].Gene[j];
		}
	}
}
/*------------------------------------------------------*/
/*              COPY_GENOTYPES                          */
/* 		Coppies a genotype to another                   */
/*										                */
void copy_genotypes(struct genotype* oldgenotype, struct genotype* newgenotype)
{
	int i=0;

	for (i=0; i<DIMENSION; i++)
	  newgenotype->Gene[i]=oldgenotype->Gene[i];
	newgenotype->Fitness=oldgenotype->Fitness;

	for (i=0; i<DIMENSION; i++)
	  newgenotype->Upper[i]=oldgenotype->Upper[i];
	for (i=0; i<DIMENSION; i++)
	  newgenotype->Lower[i]=oldgenotype->Lower[i];

	newgenotype->RFitness=oldgenotype->RFitness;
	newgenotype->CFitness=oldgenotype->CFitness;
	newgenotype->Selection_Prob=oldgenotype->Selection_Prob;
	newgenotype->Cumulative_Prob=oldgenotype->Cumulative_Prob;

	newgenotype->Survivor=oldgenotype->Survivor;
	newgenotype->Mate=oldgenotype->Mate;
	newgenotype->Mutate=oldgenotype->Mutate;
}
/*------------------------------------------------------*/
/*              COPY_POPULATION                         */
/* 		Copies a population to another population       */
/*                                                      */
void copy_population(struct genotype old_pop[POPSIZE+1],struct genotype new_pop[POPSIZE+1])
{
	int mem=0;

	for (mem=0; mem<=POPSIZE; mem++)
	  copy_genotypes(&old_pop[mem],&new_pop[mem]);
}
/*------------------------------------------------------*/
/*              SELECT                                  */
/* This is an implementation of STANDARD PROPORTIONAL SELECTION */
/* (or ROULETTE WHEEL SELECTION) for MAXIMIZATION problems      */
/* It also checks to make sure that the best member survives    */
/* (i.e., elitest selection).                                   */
/*                                                              */
void select(void)
{
	int member, spin_num, mem;
	double Total_Fitness;       /* The total population fitness */
	double roulette=0;
	/* a variable for temporary storing of the population */
	struct genotype Buffered_Pop[POPSIZE+1];
	int Found;                    /* A flag */

	/* First, find the total fitness of the population    */
	Total_Fitness=0;
	for (member=0; member<POPSIZE; member++)
	  Total_Fitness += Population[member].Fitness;

	/* Next, calculate the probability of a selection of each genotype      */
	for (member=0; member<POPSIZE; member++)
	  Population[member].Selection_Prob = Population[member].Fitness/Total_Fitness;

	/* Now, calculate the cumulative probability of each genotype     */
	Population[0].Cumulative_Prob=Population[0].Selection_Prob;

	for (member=1; member<POPSIZE; member++)
	  Population[member].Cumulative_Prob = Population[member-1].Cumulative_Prob +
	Population[member].Selection_Prob;

	/* Finally, select the survivors using the cumulative probability */
	/* and create the new Population                                  */
	for (spin_num=0; spin_num<POPSIZE; spin_num++)
	{
		Found = FALSE;
		roulette = (double) rand()/RAND_MAX;

		if (roulette < Population[0].Cumulative_Prob) 			/* Select the first member of the Population */
		  copy_genotypes(&Population[0],&Buffered_Pop[spin_num]);
		else if (roulette >Population[POPSIZE-1].Cumulative_Prob) /* select the best member of the population */
		  copy_genotypes(&Population[POPSIZE],&Buffered_Pop[spin_num]);
		else
		  for (mem=1; mem<POPSIZE && Found==FALSE; mem++)
		    if ( (roulette > Population[mem-1].Cumulative_Prob) && (roulette <=Population[mem].Cumulative_Prob) )
		    {
			  copy_genotypes(&Population[mem],&Buffered_Pop[spin_num]);
			  Found = TRUE;
		    }
	}

	/* copy the best individual */
	copy_genotypes(&Population[POPSIZE],&Buffered_Pop[POPSIZE]);

	/* Copy the Buffered_Pop to the original Population */
	copy_population(Buffered_Pop,Population);
}
/*------------------------------------------------------*/
/*              CROSSOVER                               */
/* This is an implementation of STANDARD SINGLE POINT CROSSOVER. */
/*                                                      */
void crossover(void)
{
	int mem,i,
	parent1,          /* Parent one           */
	parent2,          /* Parent two           */
	xover_point,      /* Crossover point      */
	count=0,
	lovers=0,         /* number of matting genotypes */
	indiv=0;
	struct genotype temp_parent;

	for (mem=0; mem<=POPSIZE; mem++)
	  Population[mem].Mate=FALSE;

	/* first find the individuals for the Crossover operation */
	for (mem=0;mem<POPSIZE;mem++)
	  if (frand() < PXOVER)
	  { /* frand returns a random number in the range [0,1] */
	    Population[mem].Mate = TRUE;
	    lovers++;
      }

    /* We want an even number of "lovers"*/
    if ((lovers%2) != 0)
	{
	  do  	/* find an non "lover" */
	    indiv=rand()%POPSIZE;
	  while (Population[indiv].Mate==TRUE);
	  /* make it "lover"    */
	  Population[indiv].Mate=TRUE;
	  lovers++;
    }

    /* second mate the "lovers" */
    mem=0;
    for (count=0; count<(lovers/2); count++)
	{
	  while (Population[mem].Mate==FALSE) mem++; /* find the first parent */
	  parent1=mem;
	  mem++;
	  while (Population[mem].Mate==FALSE) mem++; /* find the second parent */
	  parent2=mem;
	  mem++;

	  /* select the crossover point :1...DIMENSION-1 */
	  xover_point=(rand()%(DIMENSION-1))+1;

	  /* Perform the crossover */
	  /* copy parent1 to temp_parent */
	  copy_genotypes(&Population[parent1],&temp_parent);

	  for (i=xover_point; i<DIMENSION; i++)
	    Population[parent1].Gene[i]=Population[parent2].Gene[i];
	  for (i=xover_point;i<DIMENSION;i++)
	    Population[parent2].Gene[i]=temp_parent.Gene[i];
    }
    /* set Mate flag to False */
    for (mem=0;mem<=POPSIZE;mem++)
      Population[mem].Mate=FALSE;
}
/*------------------------------------------------------*/
/*              MUTATION                                */
/* This is an implementation of random mutation. A value selected  */
/* for mutation is replaced by a randomly generated number between */
/* that variable's lower and upper bounds.                         */
/*                                                      */
void mutate(void)
{
	double lbound,hbound;
	int member,             /* The member to be mutated                 */
	var;                /* The variable to be mutated               */

	for (member=0; member<POPSIZE; member++) /* for every member */
	  for (var=0; var<DIMENSION; var++)		/* for every gene   */
	    if (frand()<PMUTATION)
		{
		  lbound=Population[member].Lower[var];
		  hbound=Population[member].Upper[var];

		/* Generate a new random value between the bounds */
		  Population[member].Gene[var]=RandVal(lbound,hbound);
	    }
}
/*------------------------------------------------------*/
/*              REPORT                                  */
/* Report progress of the simulation. Data is dumped to a log   */
/* file in comma separeted value format which can be imported   */
/* and graphed using any commercial spreadsheet package.        */
/*                                                      */
void report(void)
{
	double best_val,      /* Best population fitness                      */
	avg,           /* Average population fitness                   */
	stddev,        /* Std. deviation of population fitness         */
	sum_square,    /* Sum of the squares for std. dev calc.        */
	square_sum,    /* Square of the sums for std. dev. calc.       */
	sum;           /* The summed population fitness                */
	int i=0;	/* counter */

	sum=0.0;
	sum_square=0.0;

	/* Calculate the summed population fitness and the sum        */
	/* of the squared individual fitnesses.                       */
	for (i=0; i<POPSIZE; i++)
	{
		sum += (Population[i].Fitness);
		sum_square += pow(Population[i].Fitness, 2);
	}

	/* Calculate the average and standard deviations of the       */
	/* population's fitness.                                      */
	avg=sum/(double)POPSIZE;
	square_sum=sum*sum/(double) POPSIZE;
	stddev=sqrt((1.0/(double)(POPSIZE-1))*(sum_square-square_sum));
	best_val=1/Population[POPSIZE].Fitness;

	/* Print the generation, best, avg, and std. deviation to a   */
	/* file in csv format.                                        */
	fprintf(results,"\n%10d  %15.6f  %15.6f  %15.6f", Generation, best_val, avg, stddev);
	/* Print the Best Genotype */
	fprintf(results, "  (");
	for (i=0;i<DIMENSION;i++)
	  fprintf(results," %5.6f ",Population[POPSIZE].Gene[i]);
	fprintf(results, ") ");
}
