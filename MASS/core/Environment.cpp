#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "dart/collision/bullet/bullet.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

Environment::
Environment()
	:mControlHz(30),
	mSimulationHz(900),
	mWorld(std::make_shared<World>()),
	mUseMuscle(true),
	w_q(0.65),w_v(0.1),w_ee(0.15),w_com(0.1)
{

}

void
Environment::
Initialize(const std::string& meta_file,bool load_obj)
{
	std::ifstream ifs(meta_file);
	if(!(ifs.is_open()))
	{
		std::cout<<"Can't read file "<<meta_file<<std::endl;
		return;
	}
	std::string str;
	std::string index;
	std::stringstream ss;
	MASS::Character* character = new MASS::Character();
	while(!ifs.eof())
	{
		str.clear();
		index.clear();
		ss.clear();

		std::getline(ifs,str);
		ss.str(str);
		ss>>index;
		if(!index.compare("use_muscle"))
		{	
			std::string str2;
			ss>>str2;
			if(!str2.compare("true"))
				this->SetUseMuscle(true);
			else
				this->SetUseMuscle(false);
		}
		else if(!index.compare("con_hz")){
			int hz;
			ss>>hz;
			this->SetControlHz(hz);
		}
		else if(!index.compare("sim_hz")){
			int hz;
			ss>>hz;
			this->SetSimulationHz(hz);
		}
		else if(!index.compare("skel_file")){
			std::string str2;
			ss>>str2;

			character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);
		}
		else if(!index.compare("muscle_file")){
			std::string str2;
			ss>>str2;
			if(this->GetUseMuscle())
				character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
		}
		else if(!index.compare("bvh_file")){
			std::string str2,str3;

			ss>>str2>>str3;
			bool cyclic = false;
			if(!str3.compare("true"))
				cyclic = true;
			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}
		else if(!index.compare("reward_param")){
			double a,b,c,d;
			ss>>a>>b>>c>>d;
			this->SetRewardParameters(a,b,c,d);

		}


	}
	ifs.close();
	
	
	double kp = 300.0;
	character->SetPDParameters(kp,sqrt(2*kp));
	this->SetCharacter(character);
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	this->Initialize();
}

void
Environment::
Initialize()
{
	if(mCharacter->GetSkeleton()==nullptr){
		std::cout<<"Initialize character First"<<std::endl;
		exit(0);
	}
	if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")
		mRootJointDof = 6;
	else if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint")
		mRootJointDof = 3;	
	else
		mRootJointDof = 0;
	mNumActiveDof = mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof;
	if(mUseMuscle)
	{
		int num_total_related_dofs = 0;
		for(auto m : mCharacter->GetMuscles()){
			m->Update();
			num_total_related_dofs += m->GetNumRelatedDofs();
		}
		mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);
		mCurrentMuscleTuple.L = Eigen::VectorXd::Zero(mNumActiveDof*mCharacter->GetMuscles().size());
		mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
		mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
		mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size());
	}
	mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
	mWorld->setTimeStep(1.0/mSimulationHz);
	mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
	mWorld->addSkeleton(mCharacter->GetSkeleton());
	mWorld->addSkeleton(mGround);
	mAction = Eigen::VectorXd::Zero(mNumActiveDof);
	
	// === Exoskeleton (hip 6-dim only) index & vector init ===
	mExoActiveIdx.clear();

	auto& skel = mCharacter->GetSkeleton();

	// Helper: given child body name, find parent joint (expect BallJoint) and map 3 axes to active dof indices
	auto collect_hip3 = [&](const std::string& child_name) -> bool {
		auto* bn = skel->getBodyNode(child_name);
		if(!bn) return false;
		auto* joint = bn->getParentJoint();
		if(!joint) return false;

		std::string jtype = joint->getType();
		std::string jtype_lower = jtype;
		std::transform(jtype_lower.begin(), jtype_lower.end(), jtype_lower.begin(), ::tolower);
		if(jtype_lower.find("ball") == std::string::npos) return false;
		if(joint->getNumDofs() != 3) return false;

		int base_full = joint->getIndexInSkeleton(0);
		int base_act  = base_full - mRootJointDof;

		if(base_act < 0 || base_act + 2 >= mNumActiveDof) return false;

		mExoActiveIdx.push_back(base_act + 0);
		mExoActiveIdx.push_back(base_act + 1);
		mExoActiveIdx.push_back(base_act + 2);
		return true;
	};

	// Collect in fixed order: left hip (FemurL) -> right hip (FemurR)
	bool okL = collect_hip3("FemurL");
	bool okR = collect_hip3("FemurR");

	// Fallback to first 6 active dofs if hip detection fails
	if(mExoActiveIdx.size() != 6){
		mExoActiveIdx.clear();
		for(int i=0;i<6 && i<mNumActiveDof;i++) mExoActiveIdx.push_back(i);
		while(mExoActiveIdx.size()<6) mExoActiveIdx.push_back(0);
	}

	// Debug print exo indices
	std::cout << "[EXO] mExoActiveIdx.size() = " << mExoActiveIdx.size() << std::endl;
	if(mExoActiveIdx.size() == 6 && okL && okR){
		std::cout << "[EXO] Using HIP 6-DoF mapping from FemurL/FemurR (BallJoint)" << std::endl;
	}else{
		std::cout << "[EXO] Fallback to first 6 active dofs." << std::endl;
	}
	for(size_t k=0; k<mExoActiveIdx.size(); ++k){
		int ai   = mExoActiveIdx[k];
		int full = mRootJointDof + ai;
		std::string dof_name = skel->getDof(full)->getName();
		std::cout << "  - active[" << ai << "] -> dof[" << full << "] = " << dof_name << std::endl;
	}

	// Exo 6-dim torque and limits (default ±120 N·m)
	mExoTau6      = Eigen::VectorXd::Zero(6);
	mExoTauLimit6 = Eigen::VectorXd::Constant(6, 120.0);
	mExoTauAct    = Eigen::VectorXd::Zero(mNumActiveDof);

	ResetExoEpisodeAccumulators();

	// Ctrl power/energy init
	mCtrlEnergyEpisode = 0.0;
	mCtrlPowerSum = 0.0;
	mCtrlPowerCount = 0;


	Reset(false);
	mNumState = GetState().rows();
}

void
Environment::
Reset(bool RSI)
{
	mWorld->reset();

	mCharacter->GetSkeleton()->clearConstraintImpulses();
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();
	
	double t = 0.0;

	if(RSI)
	t = dart::math::Random::uniform(0.0, mCharacter->GetBVH()->GetMaxTime() * 0.9);

	mWorld->setTime(t);
	mCharacter->Reset();
	mAction.setZero();

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;

	mCharacter->GetSkeleton()->setPositions(mTargetPositions);
	mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
	mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);

	// Exoskeleton: reset episode accumulators
	mExoTau6.setZero();
	mExoTauAct.setZero();
	ResetExoEpisodeAccumulators();
}

void
Environment::
Step()
{	
	if(mUseMuscle)
	{
		int count = 0;
		for(auto muscle : mCharacter->GetMuscles())
		{
			muscle->activation = mActivationLevels[count++];
			muscle->Update();
			muscle->ApplyForceToBody();
		}
		if(mSimCount == mRandomSampleIndex)
		{
			auto& skel = mCharacter->GetSkeleton();
			auto& muscles = mCharacter->GetMuscles();

			int n = skel->getNumDofs();
			int m = muscles.size();
			Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);
			Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);

			for(int i=0;i<muscles.size();i++)
			{
				auto muscle = muscles[i];
				// muscle->Update();
				Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
				auto Ap = muscle->GetForceJacobianAndPassive();

				JtA.block(0,i,n,1) = Jt*Ap.first;
				Jtp += Jt*Ap.second;
			}

			mCurrentMuscleTuple.JtA = GetMuscleTorques();
			Eigen::MatrixXd L = JtA.block(mRootJointDof,0,n-mRootJointDof,m);
			Eigen::VectorXd L_vectorized = Eigen::VectorXd((n-mRootJointDof)*m);
			for(int i=0;i<n-mRootJointDof;i++)
			{
				L_vectorized.segment(i*m, m) = L.row(i);
			}
			mCurrentMuscleTuple.L = L_vectorized;
			mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof,n-mRootJointDof);
			mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
			mMuscleTuples.push_back(mCurrentMuscleTuple);
		}
	}
	else
	{
		GetDesiredTorques();
		mCharacter->GetSkeleton()->setForces(mDesiredTorque);
	}

	// === Exoskeleton: add exo torques to generalized forces and track power/energy ===
	{
		auto& skel = mCharacter->GetSkeleton();
		const int n = skel->getNumDofs();

		Eigen::VectorXd tau_exo_full = Eigen::VectorXd::Zero(n);
		tau_exo_full.tail(mNumActiveDof) = mExoTauAct;

		skel->setForces(skel->getForces() + tau_exo_full);

		// Power/energy stats (active dofs only)
		const Eigen::VectorXd qdot_act = skel->getVelocities().tail(mNumActiveDof);
		const double P = mExoTauAct.dot(qdot_act);
		mExoPowerSum   += std::abs(P);
		mExoPowerCount += 1;
		mExoEnergyEpisode += std::abs(P) * mWorld->getTimeStep();
	}


	// Ctrl power/energy: tau_des dot qdot
	{
		const Eigen::VectorXd qdot_act = mCharacter->GetSkeleton()->getVelocities().tail(mNumActiveDof);
		const Eigen::VectorXd tau_des_act = mDesiredTorque.tail(mNumActiveDof);
		const double Pctrl = tau_des_act.dot(qdot_act);
		mCtrlPowerSum    += std::abs(Pctrl);
		mCtrlPowerCount  += 1;
		mCtrlEnergyEpisode += std::abs(Pctrl) * mWorld->getTimeStep();
	}


	mWorld->step();

	mSimCount++;
}


Eigen::VectorXd
Environment::
GetDesiredTorques()
{
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mTargetPositions.rows()-mRootJointDof) += mAction;
	mDesiredTorque = mCharacter->GetSPDForces(p_des);
	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}
Eigen::VectorXd
Environment::
GetMuscleTorques()
{
	int index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(index,JtA_i.rows()) = JtA_i;
		index += JtA_i.rows();
	}
	
	return mCurrentMuscleTuple.JtA;
}
double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}
double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
	return exp(-w*vec.squaredNorm());
}
double exp_of_squared(double val,double w)
{
	return exp(-w*val*val);
}


bool
Environment::
IsEndOfEpisode()
{
	bool isTerminal = false;
	
	Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
	Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

	double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];
	if(root_y<1.3)
		isTerminal =true;
	else if (dart::math::isNan(p) || dart::math::isNan(v))
		isTerminal =true;
	else if(mWorld->getTime()>10.0)
		isTerminal =true;
	
	return isTerminal;
}
Eigen::VectorXd 
Environment::
GetState()
{
	auto& skel = mCharacter->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getBodyNode(0);
	int num_body_nodes = skel->getNumBodyNodes();
	Eigen::VectorXd p,v;

	p.resize( (num_body_nodes-1)*3);
	v.resize((num_body_nodes)*3);

	for(int i = 1;i<num_body_nodes;i++)
	{
		p.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOM(root);
		v.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOMLinearVelocity();
	}
	
	v.tail<3>() = root->getCOMLinearVelocity();

	double t_phase = mCharacter->GetBVH()->GetMaxTime();
	double phi = std::fmod(mWorld->getTime(),t_phase)/t_phase;

	p *= 0.8;
	v *= 0.2;

	Eigen::VectorXd state(p.rows()+v.rows()+1);

	state<<p,v,phi;
	return state; 
}
void 
Environment::
SetAction(const Eigen::VectorXd& a)
{
	mAction = a*0.1;

	double t = mWorld->getTime();

	std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
	mTargetPositions = pv.first;
	mTargetVelocities = pv.second;

	mSimCount = 0;
	mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
	mAverageActivationLevels.setZero();
}

double 
Environment::
GetReward()
{
    auto& skel = mCharacter->GetSkeleton();

    Eigen::VectorXd cur_pos = skel->getPositions();
    Eigen::VectorXd cur_vel = skel->getVelocities();

    Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
    Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);

    Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
    Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());

    const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();
    for(auto ss : bvh_map)
    {
        auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();
        int idx = joint->getIndexInSkeleton(0);
        if(joint->getType()=="FreeJoint")
            continue;
        else if(joint->getType()=="RevoluteJoint"){
            p_diff[idx] = p_diff_all[idx];
            v_diff[idx] = v_diff_all[idx];
        }
        else if(joint->getType()=="BallJoint"){
            p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
            v_diff.segment<3>(idx) = v_diff_all.segment<3>(idx);
        }
    }

    auto ees = mCharacter->GetEndEffectors();
    Eigen::VectorXd ee_diff(ees.size()*3);
    Eigen::VectorXd com_diff;

    for(int i =0;i<ees.size();i++)
        ee_diff.segment<3>(i*3) = ees[i]->getCOM();
    com_diff = skel->getCOM();

    skel->setPositions(mTargetPositions);
    skel->computeForwardKinematics(true,false,false);

    com_diff -= skel->getCOM();
    for(int i=0;i<ees.size();i++)
        ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;

    skel->setPositions(cur_pos);
    skel->computeForwardKinematics(true,false,false);

    // Imitation reward
    double r_q  = exp_of_squared(p_diff,2.0);
    double r_v  = exp_of_squared(v_diff,0.1);
    double r_ee = exp_of_squared(ee_diff,40.0);
    double r_com= exp_of_squared(com_diff,10.0);

    double r = r_ee*(w_q*r_q + w_v*r_v) + w_com*r_com;

    // Muscle activation penalty + exo power penalty (for exo mode PPO)
    const double w_act_pen = 0.02;
    const double w_exo_pow = 0.001;
    const double pow_cap   = 2000.0;

    // 1) Muscle activation penalty
    double act_pen = 0.0;
    if(mUseMuscle && mActivationLevels.size() > 0){
        act_pen = mActivationLevels.squaredNorm() / (double)mActivationLevels.size();
    }

	// 2) Exo power penalty (active DOFs only)
	double exo_pow_step = 0.0;
	if(mUseExo && mExoTauAct.size() == mNumActiveDof){
		const Eigen::VectorXd qdot_active = cur_vel.tail(mNumActiveDof);
		double P = mExoTauAct.dot(qdot_active);
		exo_pow_step = std::min(std::abs(P), pow_cap);
	}

    r -= w_act_pen * act_pen;
    r -= w_exo_pow * exo_pow_step;

    return r;
}


// ===================== Exoskeleton: 6-dim methods =====================
void Environment::SetExoTorqueLimits(const Eigen::VectorXd& lim)
{
    if(lim.size() == 6)
        mExoTauLimit6 = lim;
}

void Environment::SetExoTorques(const Eigen::VectorXd& tau6)
{
    mExoTau6 = tau6;

    // Clamp
    for(int i=0; i<mExoTau6.size(); ++i){
        double limit = mExoTauLimit6[i];
        if(mExoTau6[i] >  limit) mExoTau6[i] =  limit;
        if(mExoTau6[i] < -limit) mExoTau6[i] = -limit;
    }

    // Scatter 6-dim exo torque to active dof vector
    mExoTauAct.setZero(mNumActiveDof);
    for(int i=0;i<6 && i<mExoActiveIdx.size(); ++i){
        int ai = mExoActiveIdx[i];
        if(ai >= 0 && ai < mNumActiveDof)
            mExoTauAct[ai] = mExoTau6[i];
    }
}


void Environment::ResetExoEpisodeAccumulators()
{
    mExoEnergyEpisode = 0.0;
    mExoPowerSum = 0.0;
    mExoPowerCount = 0;
}
