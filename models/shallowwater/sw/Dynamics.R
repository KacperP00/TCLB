AddDescription( "Shallow water equation", "

Lattice Boltzmann Method for Shallow Water equation on a D2Q9 lattice. 
Model has adjoint capabilities for unsteady optimization.

")


AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")
AddDensity( name="w", group="w", parameter=T)

AddQuantity( name="Rho",unit="m")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="RhoB",adjoint=T)
AddQuantity( name="UB",adjoint=T,vector=T)
AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="Gravity", default=1, comment='inlet density')
AddSetting(name="SolidH", default=1, comment='inlet density')
AddSetting(name="EnergySink", default=0, comment='inlet density')

AddSetting(name="Height", default=0, zonal=T)
AddSetting(name="EnergySink2", default=0, comment='energy sink for zone 2')

AddSetting(name="Wave_A", default=0, comment='wave amplitude')
AddSetting(name="Wave_k_real", default=0, comment='wave real wavenumber')
AddSetting(name="Wave_k_imag", default=0, comment='wave imaginary wavenumber')
AddSetting(name="Wave_w", default=0, comment='wave frequency')
AddSetting(name="Wave_Phase", default=0, comment='wave phase')
AddSetting(name="Wave_Period", default=1, comment='wave period')
AddSetting(name="Wave_Length", default=1, comment='wave length')

AddSetting(name="Adjnt_start_Obj1", default=0.0, comment='Start of adjoint algorithm for Obj1')
AddSetting(name="Adjnt_start_Obj2", default=0.0, comment='Start of adjoint algorithm for Obj2')
AddSetting(name="Adjnt_end", default=1000000000.0, comment='End of adjoint algorithm')

AddGlobal(name="PressDiff", comment='pressure loss')
AddGlobal(name="TotalDiff", comment='total variation of velocity')
AddGlobal(name="Material", comment='total material')
AddGlobal(name="EnergyGain", comment='pressure loss')
AddGlobal(name="EnergyGain2", comment='energy loss in zone 2')
AddGlobal(name="WaveError", comment='squared difference between simulated and theoretical wave')
AddGlobal(name="WaveError2", comment='measure of deviation from base height')
AddGlobal(name="Penalty", comment='Weight for pushing material to 0 or 1')

AddNodeType(name="Obj1", group="OBJECTIVE")
AddNodeType(name="Obj2", group="OBJECTIVE")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="BGK", group="COLLISION")
AddNodeType(name="MRT", group="COLLISION")
AddNodeType(name="DesignSpace", group="DESIGNSPACE")
