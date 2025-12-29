import { Container, Form, Button } from "react-bootstrap";
import { useForm } from "react-hook-form";
const SimulationModel = () => {
    const {
        register,
        handleSubmit,
      } = useForm({
        defaultValues: {
            solver_1: "base",
            solver_2: "base"
        }
      });

    return (
        <Form>
            <Container>
                <Form.Label className="fw-bold">Solver 1:</Form.Label>
                <Form.Select {...register("solver_1")}>
                    <option value="base">Base</option>
                    <option value="cosy_infinity">COSY Infinity</option>
                    <option value="xsuite">XSuite</option>
                </Form.Select>

                <Form.Label className="fw-bold">Solver 2:</Form.Label>
                <Form.Select {...register("solver_2")}>
                    <option value="base">Base</option>
                    <option value="cosy_infinity">COSY Infinity</option>
                    <option value="xsuite">XSuite</option>
                    <option value="none">None</option>
                </Form.Select>

                <Button variant="primary" type="submit" className="mt-3">
                    Run Beam Dynamics Simulation
                </Button>
            </Container>
        </Form>
    );
};

export default SimulationModel;