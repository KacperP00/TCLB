<?xml version='1.0' encoding='utf-8'?>
<CLBConfig version="2.0" output="{OUT_DIR}" permissive="false">
    <Geometry nx="{NX}" ny="{NY}">
        <MRT><Box /></MRT>
        <WPressure name="Inlet">
            <Box dx="0" nx="1" dy="{INLET_DY}" ny="{INLET_NY}" />
        </WPressure>
        <Wall name="Outlet">
            <Box dx="{OUTLET_DX}" nx="{OUTLET_NX}" />
        </Wall>
        <Wall>
            <Box dx="0" nx="{NX}" dy="{WALL_TOP_DY}" ny="{WALL_NY}" />
            <Box dx="0" nx="{EUVE_W2_NX}" dy="0" ny="{WALL_NY}" />
            <Box dx="{EUVE_C1_DX}" nx="{EUVE_C_NX}" dy="0" ny="{EUVE_C1_NY}" />
            <Box dx="{EUVE_SEP_DX}" nx="{EUVE_SEP_NX}" dy="0" ny="{WALL_NY}" />
            <Box dx="{EUVE_C2_DX}" nx="{EUVE_C_NX}" dy="0" ny="{EUVE_C2_NY}" />
            <Box dx="{EUVE_W6_DX}" nx="{EUVE_W6_NX}" dy="0" ny="{WALL_NY}" />
        </Wall>
    </Geometry>
    <Model>
        <Param name="Gravity" value="0.00209424" />
        <Param name="InletVelocity" value="0" />
        <Param name="EnergySink" value="0" />
        <Param name="EnergySink2" value="0" />
        <Param name="Height" value="9.55" />
        <Param name="nu" value="{NU}" />
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