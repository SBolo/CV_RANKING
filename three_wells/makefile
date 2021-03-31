ICPP = g++
IC = gcc
INCLUDE = -I.
OPT = -O2

OBJS = random.o committor.o Ito_integrator.o
LIBS_COMMON = -lgsl -lgslcblas -I/opt/local/include -L/opt/local/lib

main: 	$(OBJS)
	$(ICPP) $(OPT) -o ../langevin $(OBJS) $(LIBS_COMMON)

committor: 	$(OBJS)
	$(ICPP) $(OPT) -o ../committor $(OBJS) $(LIBS_COMMON)

%.o: %.cpp
	$(ICPP) $(OPT) $(DIRECT) -c $(INCLUDE) $(LIBS_COMMON) $(<) $(outputs)

clean:
	rm -fv *.o langevin
