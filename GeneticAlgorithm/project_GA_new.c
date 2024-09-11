/*--------------------------------------------------------------*
 * This is the main function. It loops for the specified number *
 * of generations. Each generation involves selecting survivors,*
 * performing crossover and mutation, and then evaluating the   *
 * resulting population.                                        */

#include "simpleGA_new.h"	/* include the sga.h header file */

int main()
{
	int i,j,k;
	char buffer;
		
	double meanDistance = 0.0;
	double meanBest[DIMENSION];
	for(i=0; i<DIMENSION; i++) meanBest[i] = 0.0;
		
	srand(time(0));

	if ((results=fopen(RESULTS_FILE,"w"))==NULL)
	{
		printf("Can't open the results file \n");
		exit(1);
	}

	for (k=1; k<(EXPERIMENTS+1); k++)
	{
		fprintf(results,"----------------------------------------------------------------\n");
		fprintf(results,"-------------------------EXPERIMENT #%d-------------------------\n", k);
		fprintf(results,"----------------------------------------------------------------\n\n");
		fprintf(results,"%10s  %15s  %15s  %15s  %20s\n", "Generation", "Best Value", "Average", "StdDev", "Best Genotype");

        Generation=0;
		initialize();
		printf("Initial Population\n");
		for (j=0; j<DIMENSION; j++)
		  printf("  Gene%d   ", j);
		printf("\n");
		for (i=0;i<POPSIZE;i++)
		{
			for(j=0;j<DIMENSION;j++)
			printf("%10.6f", Population[i].Gene[j]);
			printf("\n");
		}
		printf("-------------------------EXPERIMENT #%d-------------------------\n", k);

		evaluate();
		Population[POPSIZE].Fitness=0;
		while (Generation<MAXGENS)
		{
			Generation++;
			select();
			crossover();
			mutate();
			evaluate();
			if ( Generation % DISLPAYFREQ == 0 )
			printf("Generation : %d \t Distance = %3.6f\n",Generation, 1/Population[POPSIZE].Fitness);
			report();
		}

		printf("\n\nSimulation completed\n");
		fprintf(results,"\n\nSimulation completed\n");
		printf("\n   Best member :\n");
		fprintf(results,"\n   Best member :\n");
		for (i=0;i<DIMENSION;i++)
		{
			meanBest[i] += Population[POPSIZE].Gene[i];
			printf("\n Gene(%d) = %3.6f",i,Population[POPSIZE].Gene[i]);
			fprintf(results,"\n Gene(%d) = %3.6f",i,Population[POPSIZE].Gene[i]);
		}

		printf("\n Distance = %3.6f\n\n", (1/Population[POPSIZE].Fitness));
		fprintf(results,"\n Distance = %3.6f\n\n", (1/Population[POPSIZE].Fitness));

		meanDistance += (1/Population[POPSIZE].Fitness);
	}
	fclose(results);

	/* Output to the screen the mean best solution and the corresponding	*/
	/* mean distance							*/
	printf("\n\n\n----------------------------------------------------------------\n");
	printf("--------------------------FINAL RESULTS--------------------------\n");
	printf("----------------------------------------------------------------\n");

	printf("\n Number of Experiments         = %d", EXPERIMENTS);
	printf("\n Population size               = %d", POPSIZE);
	printf("\n Maximum number of generations = %d", MAXGENS);
	printf("\n Probability of crossover      = %2.2f", PXOVER);
	printf("\n Probability of mutation       = %2.2f", PMUTATION);

	printf("\n\n Mean Distance = %3.6f", meanDistance/EXPERIMENTS);
	printf("\n Mean Best:");
	for(i=0;i<DIMENSION;i++)
	  printf("\n\t Gene(%d) = %3.6f",i, meanBest[i]/EXPERIMENTS);
	printf("\n\n----------------------------------------------------------------\n");
	printf("--------------------------SIMULATION ENDS--------------------------\n");
	printf("----------------------------------------------------------------\n");

	return 0;
}


