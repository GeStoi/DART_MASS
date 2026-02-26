#ifndef __MASS_ENVIRONMENT_H__
#define __MASS_ENVIRONMENT_H__
#include "dart/dart.hpp"
#include "Character.h"
#include "Muscle.h"
namespace MASS
{

struct MuscleTuple
{
	Eigen::VectorXd JtA;
	Eigen::VectorXd L;
	Eigen::VectorXd b;
	Eigen::VectorXd tau_des;
};
class Environment
{
public:
	Environment();

	void SetUseMuscle(bool use_muscle){mUseMuscle = use_muscle;}
	void SetControlHz(int con_hz) {mControlHz = con_hz;}
	void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

	void SetCharacter(Character* character) {mCharacter = character;}
	void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

	void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
	void Initialize();
	void Initialize(const std::string& meta_file,bool load_obj = false);
public:
	void Step();
	void Reset(bool RSI = true);
	bool IsEndOfEpisode();
	Eigen::VectorXd GetState();
	void SetAction(const Eigen::VectorXd& a);
	double GetReward();

	Eigen::VectorXd GetDesiredTorques();
	Eigen::VectorXd GetMuscleTorques();

	// Exoskeleton interface (6-dim)
	void SetUseExo(bool use){ mUseExo = use; }
	bool GetUseExo() const { return mUseExo; }
	void SetExoTorqueLimits(const Eigen::VectorXd& lim);      // size = 6
	void SetExoTorques(const Eigen::VectorXd& tau_exo6);      // size = 6
	const Eigen::VectorXd& GetExoTorques() const { return mExoTau6; }
	double GetEpisodeExoEnergy() const { return mExoEnergyEpisode; }
	double GetEpisodeExoAvgPower() const { return (mExoPowerCount>0)? (mExoPowerSum/mExoPowerCount) : 0.0; }
	void   ResetExoEpisodeAccumulators();


	// Ctrl power/energy stats
	double GetEpisodeCtrlEnergy() const { 
    return mCtrlEnergyEpisode; 
	}
	double GetEpisodeCtrlAvgPower() const {
		return (mCtrlPowerCount>0)? (mCtrlPowerSum/mCtrlPowerCount) : 0.0;
	}

	const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
	Character* GetCharacter(){return mCharacter;}
	const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
	int GetControlHz(){return mControlHz;}
	int GetSimulationHz(){return mSimulationHz;}
	int GetNumTotalRelatedDofs(){return mCurrentMuscleTuple.JtA.rows();}
	std::vector<MuscleTuple>& GetMuscleTuples(){return mMuscleTuples;};
	int GetNumState(){return mNumState;}
	int GetNumAction(){return mNumActiveDof;}
	int GetNumSteps(){return mSimulationHz/mControlHz;}
	
	const Eigen::VectorXd& GetActivationLevels(){return mActivationLevels;}
	const Eigen::VectorXd& GetAverageActivationLevels(){return mAverageActivationLevels;}
	void SetActivationLevels(const Eigen::VectorXd& a){mActivationLevels = a;}
	bool GetUseMuscle(){return mUseMuscle;}
private:
	dart::simulation::WorldPtr mWorld;
	int mControlHz,mSimulationHz;
	bool mUseMuscle;
	Character* mCharacter;
	dart::dynamics::SkeletonPtr mGround;
	Eigen::VectorXd mAction;
	Eigen::VectorXd mTargetPositions,mTargetVelocities;

	int mNumState;
	int mNumActiveDof;
	int mRootJointDof;

	Eigen::VectorXd mActivationLevels;
	Eigen::VectorXd mAverageActivationLevels;
	Eigen::VectorXd mDesiredTorque;
	std::vector<MuscleTuple> mMuscleTuples;
	MuscleTuple mCurrentMuscleTuple;
	int mSimCount;
	int mRandomSampleIndex;

	// Exoskeleton (hip 6-dim only)
	bool mUseExo = false;              // disabled by default; Python --mode exo enables it
	std::vector<int> mExoActiveIdx;     // mapping to active dofs indices (length 6)
	Eigen::VectorXd mExoTau6;           // 6-dim exo torque (left/right hip, 3 axes each)
	Eigen::VectorXd mExoTauLimit6;      // 6-dim upper limit
	Eigen::VectorXd mExoTauAct;         // size = mNumActiveDof, scattered to active joints (used in Step())

	double mExoEnergyEpisode = 0.0;     // episode cumulative exo energy
	double mExoPowerSum = 0.0;          // cumulative abs power for avg calculation
	int    mExoPowerCount = 0;          // count


		// Ctrl power/energy
	double mCtrlEnergyEpisode = 0.0;
	double mCtrlPowerSum = 0.0;
	int    mCtrlPowerCount = 0;

	double w_q,w_v,w_ee,w_com;
};
};

#endif
