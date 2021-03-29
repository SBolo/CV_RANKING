#include "random.h"

#include <iostream>
#include <vector>

using namespace std;

class Ito_integrator {

    public:
        Ito_integrator( RandomNumbers *, int, double, double, double, double, vector<double> * , vector<double> *);
        Ito_integrator( RandomNumbers *, int, int, double, double, double, double, double, double, vector<double> *,  vector<double> *, vector<vector<double> >* );
        Ito_integrator( RandomNumbers *, int, double, double, double, vector<double> * );
        ~Ito_integrator();

        void evolve();
        void load_mpath(vector<vector<double> >* , int);

        vector<vector<double> >* get_mpath();
        double get_position( int );
        double get_z();
        double get_z_min();
        double get_s();
        double get_s_min();
        double get_w();
        double get_w_min();
        double get_penalty();
        double get_force();
        double get_bforce();
        void calc_z();
        void calc_rcs_and_grads();
        void test( int );
        void calc_sigma();



    private:

        string method;

        RandomNumbers *rng;

        vector<double> *position;
        vector<double> *grad_s;
        vector<double> *grad_w;
        vector<double> *target;
        vector<double> *f;
        vector<double> *b;

        vector<vector<double> > *mpath;

        int dim;

        double dt;
        double adim_dt;
        double variance;
        double tot_time;
        double z;
        double z_min;
        double s;
        double s_min;
        double w;
        double w0;
        double w_min; //PERHAPS IT IS JUST 0!
        double time;
        double kR;
        double ks;
        double kw;
        double lambda;
        double kbT;
        double penalty;
        double force_mod;
        double bforce_mod;

        void force();

        double d_hypot( vector<double> * );
        double potential();

};
