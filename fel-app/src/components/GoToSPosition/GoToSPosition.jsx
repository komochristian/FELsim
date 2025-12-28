import { Container, Form, Button } from "react-bootstrap";
import { useForm } from "react-hook-form";

const GoToSPosition = ({ goToSPos }) => {
    const {
        register,
        handleSubmit,
      } = useForm({
        defaultValues: {
          s_pos: 0.0,
        }
      });

  return (
    <Container>
        <Form onSubmit={handleSubmit(goToSPos)}>
            <Form.Group className="mb-3" controlId="sPositionInput">
                <Form.Label>Enter S Position (m):</Form.Label>
                <Form.Control 
                    type="number" 
                    step="any"
                    {...register("s_pos", { required: true })} 
                />
            </Form.Group>
            <Button variant="primary" type="submit">
                Go
            </Button>
        </Form>
    </Container>
  );
}

export default GoToSPosition;