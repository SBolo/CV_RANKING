#include "Ito_integrator.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <string>

using namespace std;

#define MAX 1.e50

/** CONSTRUCTOR FOR rMD **/
Ito_integrator::Ito_integrator( RandomNumbers *r, int dmn, double tmstp, double gamma, double kT, double k_ratchet, vector<double> *x0, vector<double> *t ) {

    method = "rMD";

    rng = r;

    dt = tmstp;
    adim_dt = (dt/gamma);
    variance = sqrt( 2. * kT * adim_dt );

    dim = dmn;
    time = 0.;
    kR = k_ratchet;
    penalty = 0.;

    position = new vector<double>(dim);
    target = new vector<double>(dim);
    f = new vector<double>(dim);
    b = new vector<double>(dim);


    for( int d = 0; d < dim; d++ ) {
        (*position)[d] = (*x0)[d];
        (*target)[d] = (*t)[d];
    }


    calc_z();
    z_min = z;

}

Ito_integrator::Ito_integrator( RandomNumbers *r, int dmn, double tmstp, double gamma, double kT, vector<double> *x0 ) {

    method = "MD";
    rng = r;

    dt = tmstp;
    adim_dt = (dt/gamma);
    variance = sqrt( 2. * kT * adim_dt );
    dim = dmn;
    time = 0.;

    position = new vector<double>(dim);
    f = new vector<double>(dim);
    b = new vector<double>(dim);

    for( int d = 0; d < dim; d++ ) {
        (*position)[d] = (*x0)[d];
    }
}

/** CONSTRUCTOR FOR SCPS **/
Ito_integrator::Ito_integrator( RandomNumbers *r, int dmn, int subs, double tmstp, double gamma, double kT, double k_s, double k_w, double lamb, vector<double> *x0, vector<double> *t, vector<vector<double> >* mpa ) {

    method = "SCPS";

    rng = r;

    dt = tmstp;
    adim_dt = (dt/gamma);
    variance = sqrt( 2. * kT * adim_dt );

    dim = dmn;
    time = 0.;
    ks = k_s;
    kw = k_w;
    lambda = lamb;
    penalty = 0.;

    position = new vector<double>(dim);
    target = new vector<double>(dim);
    grad_s = new vector<double>(dim);
    grad_w = new vector<double>(dim);
    f = new vector<double>(dim);
    b = new vector<double>(dim);

    mpath = new vector<vector<double> >(dim, vector<double>(subs));

    int str = ((*mpa)[0].size())/subs;

    /** LOADING MPATH, INITIAL AND FINAL POSITION **/
    for( int d = 0; d < dim; d++ ) {

        (*position)[d] = (*x0)[d];
        (*target)[d] = (*t)[d];

        for( int t = 0; t < subs; t++ ) {
            (*mpath)[d][t] = (*mpa)[d][t*str];
        }
    }

    calc_z();
    z_min = z;

    calc_rcs_and_grads();
    s_min = s;
    w_min = w;

}

//------------------------------------------------------------------------------


Ito_integrator::~Ito_integrator() {

    delete position;
    delete target;
    delete f;
    delete b;
    if(method=="SCPS") {
        delete grad_s;
        delete grad_w;
        delete mpath;
    }

}
//------------------------------------------------------------------------------

double Ito_integrator::d_hypot( vector<double> *vec ) {

    double tmp = 0;
    for( int d = 0; d < dim; d++ ) {
        tmp += (*vec)[d] * (*vec)[d];
    }

    return sqrt(tmp);
}
//------------------------------------------------------------------------------

void Ito_integrator::calc_z() {

    double dist = 0, pos, trg;

        for( int d = 0; d < dim; d++ ) {
            pos =(*position)[d];
            trg = (*target)[d];
            dist += ( pos - trg ) * ( pos - trg );
        }

    z = sqrt(dist);
}

//------------------------------------------------------------------------------
//It is convenient to simultaneously compute s and w but also their gradients. In this way you don't have to evaluate integrals too many times!

void Ito_integrator::calc_rcs_and_grads() {

    int ttot = (*mpath)[0].size();
    double dist, pos, mp, attempt;
    double num =0.;
    double den =0.;
    vector<double>* temp1 = new vector<double>(dim);
    vector<double>* temp2 = new vector<double>(dim);


    for(int t=0; t< ttot; t++){

        dist=0;
        //This cycle is to compute distance^2
        for( int d = 0; d < dim; d++ ) {
            pos = (*position)[d];
            mp = (*mpath)[d][t];
            dist += ( pos - mp ) * ( pos - mp );
        }

        num += t * exp(-lambda*dist);
        den += exp(-lambda*dist);


        //These quantities are needed in the gradient computation
        for( int d = 0; d < dim; d++ ) {

            pos = (*position)[d];
            mp = (*mpath)[d][t];
            (*temp1)[d] += 2. * (pos - mp) * t * exp(-lambda*dist);
            (*temp2)[d] += 2. * (pos - mp) * exp(-lambda*dist);
        }

    }


    s = 1. - num/(ttot*den);


    //HERE'S THE PROBLEM!!!!!!
    attempt = -  log(den)/lambda;
    //if( isnan(attempt) == 0 )
    w = attempt; //There should be a dt here, however this would produce a constant term which simplifies in the quantity w - wmin.


    for( int d = 0; d < dim; d++ ) {
        (*grad_s)[d] = ( (*temp1)[d] * den - num * (*temp2)[d] ) * lambda / (den * den * ttot);
        (*grad_w)[d] = (*temp2)[d]/den;
    }

    delete temp1;
    delete temp2;
}
//------------------------------------------------------------------------------

void Ito_integrator::test( int label ) {

    int ttot = (*mpath)[0].size();
    double dist, pos, mp, temp;
    double den =0.;
    char filename[50];
    vector <double> counter;
    ofstream output;

    for(int t=0; t< ttot; t++){

        dist=0.;
        //This cycle is to compute distance^2
        for( int d = 0; d < dim; d++ ) {
            pos = (*position)[d];
            mp = (*mpath)[d][t];
            dist += ( pos - mp ) * ( pos - mp );
        }

        temp = exp(-lambda*dist);
        den += temp;
        counter.push_back(temp);

    }

    //PRINT HOW MANY TERMS INFLUENCE THE INTEGRAL
    sprintf(filename, "trajs/relative_influence_%d.txt", label);
    output.open(filename, ios::out);

    for (int i=0; i< counter.size(); i++) {
        output << counter[i]/den <<endl;
    }
    output.close();

}
//------------------------------------------------------------------------------

double Ito_integrator::get_position( int i ) {

    if( i > dim ) {
        return 0;
    } else {
        return (*position)[i];
    }
}
//------------------------------------------------------------------------------

double Ito_integrator::get_z() {
    return z;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_z_min() {
    return z_min;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_s() {
    return s;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_s_min() {
    return s_min;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_w() {
    return w;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_w_min() {
    return w_min;
}
//------------------------------------------------------------------------------

/* MUST BE DECIDED IF TO IMPLEMENT ANOTHER FUNCTION TO RETURN THE OTHER R C, DEPENDING IF THE TWO SIGMA AND W ARE NEEDED OR JUST SIGMA */

void Ito_integrator::force() {

    /*
     * -gradV(x) where V(x) is the three-weels potential MODIFIED!!!!!!
     * Analytical form and parameters were taken from
     * Philipp Metznera, Christof Schütteb, Eric Vanden-Eijnden "Illustration of transition path
     * theory on a collection of simple examples"
     */

    double x = (*position)[0];
    double y = (*position)[1];

    double x2 = x*x;
    double xm1 = (x-1)*(x-1);
    double xp1 = (x+1)*(x+1);
    double y2 = y*y;
    double yc = (y-5./3.)*(y-5./3.);
    double yu = (y-1./3.)*(y-1./3.);

    (*f)[0] = -10*(x-1)*exp(-xm1 -y2) + x*exp(-x2)*( -6*exp(-yc) + 10*exp(-y2)) - 10*(x+1)*exp(-xp1-y2) - 0.8*x*x2;
    (*f)[1] = -6*(y-5./3.)*exp(-x2-yc) + 10*y*exp(-x2-y2) - 10*y*exp(-y2)*(exp(-xm1)+exp(-xp1)) - 0.8*yu*(y-1./3.);

}
//------------------------------------------------------------------------------

void Ito_integrator::evolve() {

    double tmp;

    force();

    if( method == "rMD" ){

        /** UPDATING RC **/
        calc_z();
        if(z < z_min) {
            z_min = z;
        }

        for( int d = 0; d < dim; d++ ) {
            (*b)[d] = - (kR/2.) * ( ((*position)[d] - (*target)[d])/z ) * ( z - z_min );
            (*position)[d] += ( (*f)[d] + (*b)[d] ) * adim_dt + variance * rng->gaussian(1.);
        }

    } else if( method == "SCPS") {

        /** UPDATING RC **/
        calc_z();
        calc_rcs_and_grads();

        if(z < z_min) {
            z_min = z;
        }

        if(s < s_min) {
            s_min = s;
        }
        if(w < w_min) {
            w_min = w;
        }

        for( int d = 0; d < dim; d++ ) {
            (*b)[d] = - (ks/2.) * (*grad_s)[d] * ( s - s_min )  - (kw/2.) * (*grad_w)[d] * ( w - w_min );
            (*position)[d] += ( (*f)[d] + (*b)[d] ) * adim_dt + variance * rng->gaussian(1.);

        }
    } else if( method == "MD" ) {
        for( int d = 0; d < dim; d++ ) {
            (*position)[d] += (*f)[d] * adim_dt + variance * rng->gaussian(1.);
        }

    }

    tmp = d_hypot(b);
    tmp *= tmp;

    penalty += tmp;
    force_mod = d_hypot(f);
    bforce_mod = tmp;

    force_mod *= force_mod;

    time += dt;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_penalty() {
    return penalty;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_force() {
    return force_mod;
}
//------------------------------------------------------------------------------

double Ito_integrator::get_bforce() {
    return bforce_mod;
}
//------------------------------------------------------------------------------

//----> WILL BECOME NECESSARY TO LOAD ALL MEAN PATHS, TO COMPUTE SIGMA!
/*
void Ito_integrator::load_mpath(vector<vector<double> >* mpa, int subs){


    int str = ((*mpa)[0].size())/subs;
    for( int d = 0; d < dim; d++ ) {

        //LOADING INITIAL POSITION (WHICH IS X0 + A FLUCTUATION
        (*position)[d] = (*mpa)[d][0];

        for( int t = 0; t < subs; t++ ) {
            (*mpath)[d][t] = (*mpa)[d][t*str];

        }
    }

}
*/
//------------------------------------------------------------------------------

vector<vector<double> >* Ito_integrator::get_mpath(){

    return mpath;
}

//------------------------------------------------------------------------------
