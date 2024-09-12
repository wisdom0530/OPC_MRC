import sys
sys.path.append(".")
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pylitho.simple as lithosim
# import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

class SimpleCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]

class SimpleILT: 

    def __init__(self, config=SimpleCfg("./config/simpleilt2048.txt"), lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False): 
        super(SimpleILT, self).__init__()
        self._config = config
        self._device = device
        # Lithosim
        self._lithosim = lithosim.to(DEVICE)
        if multigpu: 
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
    


    def objective(self):

        x = r = None
        for p1, p2 in zip(
                self.parameters(),
                self.meta_parameters(),
        ):
            x = p1
            r = p2
            break

        # symmetry control for x
        x = self.symmetry_control(x)

        # two phase projection
        x = self.active_func(x)
        y = two_phase_projection(x, d_s, d_v)

        # symmetry control for r
        r = self.symmetry_control(r)
        # r = 0.5 * tanh(r)
        # remove dim
        r = torch.squeeze(r)
        x = torch.squeeze(x)

        similarity = -torch.mean(torch.multiply(r, y))
        penalty = torch.mean((1 - torch.abs(2 * y - 1)))
        loss = similarity + self.tao * penalty

        self.c_vio = penalty.data

        return loss


    def two_phase_projection(x, d_s, d_v):
        """
        Two-phase projection
        :param x: design variable
        :param d_s: solid diameter
        :param d_v: void diameter
        :return: y: element density
        """
        alpha_s = torch.FloatTensor([0.001])
        alpha_v = torch.FloatTensor([0.001])
        beta = torch.FloatTensor([6])

        n_s = -1 * (torch.log(alpha_s))
        n_v = -1 * (torch.log(alpha_v))

        x_s = (1 + alpha_s) / (1 + alpha_s * torch.exp(2 * n_s * (1 - x)))
        x_v = -(1 + alpha_v) / (1 + alpha_v * torch.exp(2 * n_v * (1 + x)))

        h_s = create_conic_filter(d_s)
        h_v = create_conic_filter(d_v)

        if len(x_s.shape) < 4:
            x_s = torch.unsqueeze(torch.unsqueeze(x_s, dim=0), dim=0)

        if len(x_v.shape) < 4:
            x_v = torch.unsqueeze(torch.unsqueeze(x_v, dim=0), dim=0)

        mu_s = torch.conv2d(x_s, h_s, padding='same')
        mu_v = torch.conv2d(x_v, h_v, padding='same')

        rho_s = 1 - torch.exp(-beta * mu_s) + mu_s * torch.exp(-beta)
        rho_v = -1 + torch.exp(beta * mu_v) + mu_v * torch.exp(-beta)

        y = (rho_s + (1 + rho_v)) / 2
        y = torch.squeeze(y)

        return y


    def create_conic_filter(diameter):

        if diameter % 2 == 0:
            raise ValueError("Diameter should be an odd number.")

        kernel = np.zeros((diameter, diameter))

        center = diameter // 2
        max_distance = np.sqrt(2 * (center ** 2))

        for i in range(diameter):
            for j in range(diameter):
                distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                value = max(0, (1 - distance / max_distance))
                kernel[i, j] = 1

        kernel = kernel / np.sum(kernel)
        kernel = torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(kernel), dim=0), dim=0)

        return kernel




    def solve(self, target, params, curv=None, verbose=1): 
        # Initialize
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor): 
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        opt = optim.SGD([params], lr=self._config["StepSize"])
        # opt = optim.Adam([params], lr=self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None


        for idx in range(self._config["Iterations"]): 


            mask = torch.sigmoid(self._config["SigmoidSteepness"] * params) * self._filter
            mask += torch.sigmoid(self._config["SigmoidSteepness"] * backup) * (1.0 - self._filter)
            printedNom, printedMax, printedMin = self._lithosim(mask)
            

            l2loss = func.mse_loss(printedNom, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))
            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss


            if not curv is None: 
                kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
                curvature = func.conv2d(mask[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
                losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")
                loss += curv * losscurv
            if verbose == 1: 
                #print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")
                print(f"[Iteration {idx}]: Loss = {loss.item():.0f}; L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                bestMask = mask.detach().clone()
            
            opt.zero_grad()
            loss.backward()
            opt.step()


            for idx in range(50):

                x = tuple(params())
                inner_optim = optim.Adam(x, lr=0.025)
                loop_time = 12
                self.tao = 0.25
                with torch.enable_grad():
                    for i in range(loop_time):
                        for _ in range(6):

                            loss = self.objective()

                            inner_optim.zero_grad()
                            loss.backward(inputs=x)
                            inner_optim.step()

                    #self.tao *= 3.6




        
        return l2Min, pvbMin, bestParams, bestMask


def parallel(): 
    SCALE = 4
    l2s = []
    pvbs = []
    epes = []
    shots = []
    targetsAll = []
    paramsAll = []
    cfg   = SimpleCfg("./config/simpleilt512.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho, multigpu=True)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)
    shotCount = evaluation.ShotCounter(litho, 0.5)
    for idx in range(1, 11): 
        print(f"[SimpleILT]: Preparing testcase {idx}")
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        targetsAll.append(torch.unsqueeze(target, 0))
        paramsAll.append(torch.unsqueeze(params, 0))
    count = torch.cuda.device_count()
    print(f"Using {count} GPUs")
    while count > 0 and len(targetsAll) % count != 0: 
        targetsAll.append(targetsAll[-1])
        paramsAll.append(paramsAll[-1])
    print(f"Augmented to {len(targetsAll)} samples. ")
    targetsAll = torch.cat(targetsAll, 0)
    paramsAll = torch.cat(paramsAll, 0)

    begin = time.time()
    l2, pvb, bestParams, bestMask = solver.solve(targetsAll, paramsAll)
    runtime = time.time() - begin

    for idx in range(1, 11): 
        mask = bestMask[idx-1]
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb = test.run(mask, target, scale=SCALE)
        epeIn, epeOut = epeCheck.run(mask, target, scale=SCALE)
        epe = epeIn + epeOut
        shot = shotCount.run(mask, shape=(512, 512))
        cv2.imwrite(f"./tmp/MOSAIC_test{idx}.png", (mask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.0f}; Shot {np.mean(shots):.0f}; SolveTime {runtime:.2f}s")


def serial(): 
    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    cfg   = SimpleCfg("./config/simpleilt2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = SimpleILT(cfg, litho)
    for idx in range(1, 11): 
        design = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=SCALE)
        design.center(cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"], cfg["TileSizeY"], cfg["OffsetX"], cfg["OffsetY"])
        
        begin = time.time()
        l2, pvb, bestParams, bestMask = solver.solve(target, params, curv=None)
        runtime = time.time() - begin
        
        ref = glp.Design(f"./benchmark/ICCAD2013/M1_test{idx}.glp", down=1)
        ref.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(ref, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=True)
        cv2.imwrite(f"./tmp/MOSAIC_test{idx}.png", (bestMask * 255).detach().cpu().numpy())

        print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; SolveTime: {runtime:.2f}s")

        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        shots.append(shot)
        runtimes.append(runtime)
    
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; Shot {np.mean(shots):.1f}; SolveTime {np.mean(runtimes):.2f}s")


if __name__ == "__main__": 
    serial()
    # parallel()
