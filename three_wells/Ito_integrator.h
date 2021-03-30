#include "random.h"

#include <iostream>
#include <vector>

using namespace std;

/* Class for the simulation of overdamped Langevin dynamics in a given number of dymensions */
class Ito_integrator {

    public:
        /*
         * Public functions provided:
         * Ito_integrator: constructor
         * ~Ito_integrator: destructor
         * evolve: system's time evolution
         * calc_z: calculation fo the distance from a reference point
         * get_position: returns the i-th coordinate of the position
         * get_force: returns the squared modulus of the force
         */
        Ito_integrator(RandomNumbers *, int, double, double, double, vector<double>);
        ~Ito_integrator();

        void evolve();
        double calc_z(vector<double>);
        double get_position(int);
        double get_force();

    private:
        RandomNumbers *rng;

        vector<double> position;
        vector<double> f;
        int dim;

        double dt;
        double adim_dt;
        double variance;
        double tot_time;
        double time;
        double kbT;
        double force_mod;

        void force();
        double d_hypot(vector<double>);

};
