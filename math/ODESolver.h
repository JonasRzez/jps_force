/**
 * File:   ODESolver.h
 *
 * Created on 17. August 2010, 15:31
 *
 * @section LICENSE
 * This file is part of JuPedSim.
 *
 * JuPedSim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * JuPedSim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with JuPedSim. If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 *
 *
 *
 */

#ifndef _ODESOLVER_H
#define	_ODESOLVER_H

#include "ForceModel.h"
#include "../geometry/Building.h"

#include <vector>
using namespace std;

class ODESolver {
protected:
    ForceModel *model;
public:
    ODESolver(ForceModel* model);
    virtual void solveODE(double t, double tp, Building* building) const = 0;
};

// Euler Löser für die Differentialgleichung

class EulerSolver : public ODESolver {
public:
    EulerSolver(ForceModel *model);
    virtual void solveODE(double t, double tp, Building* building) const;
};

// Velocity Verlet Löser für die Differentialgleichung

class VelocityVerletSolver : public ODESolver {
public:
    VelocityVerletSolver(ForceModel *model);
    virtual void solveODE(double t, double tp, Building* building) const;
};

// Leapfrog Löser für die Differentialgleichung

class LeapfrogSolver : public ODESolver {
public:
    LeapfrogSolver(ForceModel *model);
    virtual void solveODE(double t, double tp, Building* building) const;
};

// für Linked Cell (Ulrich)

class EulerSolverLC : public ODESolver {
public:
    EulerSolverLC(ForceModel *model);
    virtual void solveODE(double t, double tp, Building* building) const;
};



#endif	/* _ODESOLVER_H */
