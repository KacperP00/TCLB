<?xml version='1.0' encoding='utf-8'?>
<CLBConfig version="2.0" output="{OUT_DIR}" permissive="false">
    <Geometry nx="{NX}" ny="{NY}">
        <MRT><Box /></MRT>
        <WPressure name="Inlet">
            <Box dx="0" nx="1" dy="{INLET_DY}" ny="{INLET_NY}" />
        </WPressure>
        <DesignSpace>
            <Box dx="{DS_DX}" nx="{DS_NX}" dy="{DS1_DY}" ny="{DS_NY}" />
            <Box dx="{DS_DX}" nx="{DS_NX}" dy="{DS2_DY}" ny="{DS_NY}" />
        </DesignSpace>
        <Wall name="Outlet">
            <Box dx="{OUTLET_DX}" nx="{OUTLET_NX}" />
        </Wall>
        <Wall>
            <Box dx="0" nx="{DS_DX}" dy="0" ny="{WALL_NY}" />
            <Box dx="{WALL_AFTER_DX}" nx="{WALL_AFTER_NX}" dy="0" ny="{WALL_NY}" />
            <Box dx="{DS_DX}" nx="{DS_NX}" dy="0" ny="{DS1_DY}" />
            <Box dx="0" nx="{DS_DX}" dy="{WALL_TOP_DY}" ny="{WALL_NY}" />
            <Box dx="{WALL_AFTER_DX}" nx="{WALL_AFTER_NX}" dy="{WALL_TOP_DY}" ny="{WALL_NY}" />
            <Box dx="{DS_DX}" nx="{DS_NX}" dy="{WALL_BACK2_DY}" ny="{DS1_DY}" />
            <STL file="{STL_FILE}" />
        </Wall>
        <Obj1 name="Transmission_Zone">
            <Box dx="{OBJ1_DX}" nx="{OBJ_NX}" dy="{OBJ_DY}" ny="{OBJ_NY}" />
        </Obj1>
        <Obj2 name="Reflection_Zone">
            <Box dx="{OBJ2_DX}" nx="{OBJ_NX}" dy="{OBJ_DY}" ny="{OBJ_NY}" />
        </Obj2>
    </Geometry>
    <Model>
        <Param name="Gravity" value="0.00209424" />
        <Param name="InletVelocity" value="0" />
        <Param name="EnergySink" value="0" />
        <Param name="EnergySink2" value="0" />
        <Param name="Height" value="9.55" />
        <Param name="WaveErrorInObj" value="-1.0" />
        <Param name="WaveError2InObj" value="-1.0" />
        <Param name="nu" value="{NU}" />
        <Param name="Adjnt_start_Obj1" value="{ADJ_START1}" />
        <Param name="Adjnt_start_Obj2" value="{ADJ_START2}" />
        <Param name="Adjnt_end" value="{ADJ_END}" />
        <Control Iterations="{SOLVE_ITERS}">
            <CSV file="{CSV_FILE}" Time="iter">
                <Param name="Height" value="h_inlet+9.55" zone="Inlet" />
            </CSV>
        </Control>
    </Model>
    <VTK Iterations="{VTK_ITERS}" what="Rho"/>
    <Log Iterations="{LOG_ITERS}"/>
    <Solve Iterations="{SOLVE_ITERS}"/>
</CLBConfig>